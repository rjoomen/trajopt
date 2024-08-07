#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
extern "C" {
#include "gurobi_c.h"
}
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_sco/gurobi_interface.hpp>
#include <trajopt_common/logging.hpp>
#include <trajopt_common/stl_to_string.hpp>

namespace sco
{
GRBenv* gEnv;

#if 0
void simplify(IntVec& inds, DblVec& vals) {
  // first find the largest element of inds
  // make an array of that size
  //

  assert(inds.size() == vals.size());
  int min_ind = *std::min_element(inds.begin(), inds.end());
  int max_ind = *std::max_element(inds.begin(), inds.end());
  int orig_size = inds.size();
  int work_size = max_ind - min_ind + 1;
  DblVec work_vals(work_size, 0);

  for (int i=0; i < orig_size; ++i) {
    work_vals[inds[i] - min_ind] += vals[i];
  }


  int i_new = 0;
  for (int i=0; i < work_size; ++i) {
    if (work_vals[i] != 0) {
      vals[i_new] = work_vals[i];
      inds[i_new] = i + min_ind;
      ++i_new;
    }
  }

  inds.resize(i_new);
  vals.resize(i_new);

}
#endif

#define ENSURE_SUCCESS(expr)                                                                                           \
  do                                                                                                                   \
  {                                                                                                                    \
    bool error = expr;                                                                                                 \
    if (error)                                                                                                         \
    {                                                                                                                  \
      printf("GRB error: %s while evaluating %s at %s:%i\n", GRBgeterrormsg(gEnv), #expr, __FILE__, __LINE__);         \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

Model::Ptr createGurobiModel()
{
  auto out = std::make_shared<GurobiModel>();
  return out;
}

GurobiModel::GurobiModel()
{
  if (!gEnv)
  {
    GRBloadenv(&gEnv, nullptr);
    if (trajopt_common::GetLogLevel() < trajopt_common::LevelDebug)
    {
      ENSURE_SUCCESS(GRBsetintparam(gEnv, "OutputFlag", 0));
    }
  }
  GRBnewmodel(gEnv, &m_model, "problem", 0, nullptr, nullptr, nullptr, nullptr, nullptr);
}

Var GurobiModel::addVar(const std::string& name)
{
  const std::scoped_lock lock(m_mutex);
  ENSURE_SUCCESS(GRBaddvar(
      m_model, 0, nullptr, nullptr, 0, -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, const_cast<char*>(name.c_str())));
  m_vars.push_back(std::make_shared<VarRep>(m_vars.size(), name, this));
  return m_vars.back();
}

Var GurobiModel::addVar(const std::string& name, double lb, double ub)
{
  const std::scoped_lock lock(m_mutex);
  ENSURE_SUCCESS(GRBaddvar(m_model, 0, nullptr, nullptr, 0, lb, ub, GRB_CONTINUOUS, const_cast<char*>(name.c_str())));
  m_vars.push_back(std::make_shared<VarRep>(m_vars.size(), name, this));
  return m_vars.back();
}

Cnt GurobiModel::addEqCnt(const AffExpr& expr, const std::string& name)
{
  const std::scoped_lock lock(m_mutex);
  LOG_TRACE("adding eq constraint: %s = 0", CSTR(expr));
  IntVec inds;
  vars2inds(expr.vars, inds);
  DblVec vals = expr.coeffs;
  simplify2(inds, vals);
  ENSURE_SUCCESS(GRBaddconstr(m_model,
                              static_cast<int>(inds.size()),
                              const_cast<int*>(inds.data()),
                              const_cast<double*>(vals.data()),
                              GRB_EQUAL,
                              -expr.constant,
                              const_cast<char*>(name.c_str())));
  m_cnts.push_back(std::make_shared<CntRep>(m_cnts.size(), this));
  return m_cnts.back();
}
Cnt GurobiModel::addIneqCnt(const AffExpr& expr, const std::string& name)
{
  const std::scoped_lock lock(m_mutex);
  LOG_TRACE("adding ineq: %s <= 0", CSTR(expr));
  IntVec inds;
  vars2inds(expr.vars, inds);
  DblVec vals = expr.coeffs;
  simplify2(inds, vals);
  ENSURE_SUCCESS(GRBaddconstr(m_model,
                              static_cast<int>(inds.size()),
                              inds.data(),
                              vals.data(),
                              GRB_LESS_EQUAL,
                              -expr.constant,
                              const_cast<char*>(name.c_str())));
  m_cnts.push_back(std::make_shared<CntRep>(m_cnts.size(), this));
  return m_cnts.back();
}
Cnt GurobiModel::addIneqCnt(const QuadExpr& qexpr, const std::string& name)
{
  const std::scoped_lock lock(m_mutex);
  int numlnz = static_cast<int>(qexpr.affexpr.size());
  IntVec linds;
  vars2inds(qexpr.affexpr.vars, linds);
  DblVec lvals = qexpr.affexpr.coeffs;
  IntVec inds1;
  vars2inds(qexpr.vars1, inds1);
  IntVec inds2;
  vars2inds(qexpr.vars2, inds2);
  ENSURE_SUCCESS(GRBaddqconstr(m_model,
                               numlnz,
                               linds.data(),
                               lvals.data(),
                               static_cast<int>(qexpr.size()),
                               inds1.data(),
                               inds2.data(),
                               const_cast<double*>(qexpr.coeffs.data()),
                               GRB_LESS_EQUAL,
                               -qexpr.affexpr.constant,
                               const_cast<char*>(name.c_str())));
  return Cnt();
}

void resetIndices(VarVector& vars)
{
  for (std::size_t i = 0; i < vars.size(); ++i)
    vars[i].var_rep->index = i;
}
void resetIndices(CntVector& cnts)
{
  for (std::size_t i = 0; i < cnts.size(); ++i)
    cnts[i].cnt_rep->index = i;
}

void GurobiModel::removeVars(const VarVector& vars)
{
  const std::scoped_lock lock(m_mutex);
  IntVec inds;
  vars2inds(vars, inds);
  ENSURE_SUCCESS(GRBdelvars(m_model, static_cast<int>(inds.size()), inds.data()));
  for (std::size_t i = 0; i < vars.size(); ++i)
    vars[i].var_rep->removed = true;
}

void GurobiModel::removeCnts(const CntVector& cnts)
{
  const std::scoped_lock lock(m_mutex);
  IntVec inds;
  cnts2inds(cnts, inds);
  ENSURE_SUCCESS(GRBdelconstrs(m_model, static_cast<int>(inds.size()), inds.data()));
  for (std::size_t i = 0; i < cnts.size(); ++i)
    cnts[i].cnt_rep->removed = true;
}

#if 0
void GurobiModel::setVarBounds(const Var& var, double lower, double upper) {
  assert(var.var_rep->creator == this);
  ENSURE_SUCCESS(GRBsetdblattrelement(m_model, GRB_DBL_ATTR_LB, var.var_rep->index, lower));
  ENSURE_SUCCESS(GRBsetdblattrelement(m_model, GRB_DBL_ATTR_UB, var.var_rep->index, upper));
}
#endif

void GurobiModel::setVarBounds(const VarVector& vars, const DblVec& lower, const DblVec& upper)
{
  assert(vars.size() == lower.size() && vars.size() == upper.size());
  IntVec inds;
  vars2inds(vars, inds);
  ENSURE_SUCCESS(GRBsetdblattrlist(
      m_model, GRB_DBL_ATTR_LB, static_cast<int>(inds.size()), inds.data(), const_cast<double*>(lower.data())));
  ENSURE_SUCCESS(GRBsetdblattrlist(
      m_model, GRB_DBL_ATTR_UB, static_cast<int>(inds.size()), inds.data(), const_cast<double*>(upper.data())));
}

#if 0
double GurobiModel::getVarValue(const Var& var) const {
  assert(var.var_rep->creator == this);
  double out;
  ENSURE_SUCCESS(GRBgetdblattrelement(m_model, GRB_DBL_ATTR_X, var.var_rep->index, &out));
  return out;
}
#endif

DblVec GurobiModel::getVarValues(const VarVector& vars) const
{
  assert((vars.size() == 0) || (vars[0].var_rep->creator == this));
  IntVec inds;
  vars2inds(vars, inds);
  DblVec out(inds.size());
  ENSURE_SUCCESS(GRBgetdblattrlist(m_model, GRB_DBL_ATTR_X, static_cast<int>(inds.size()), inds.data(), out.data()));
  return out;
}

CvxOptStatus GurobiModel::optimize()
{
  ENSURE_SUCCESS(GRBoptimize(m_model));
  int status;
  GRBgetintattr(m_model, GRB_INT_ATTR_STATUS, &status);
  if (status == GRB_OPTIMAL)
  {
    double objval;
    GRBgetdblattr(m_model, GRB_DBL_ATTR_OBJVAL, &objval);
    LOG_DEBUG("solver objective value: %.3e", objval);
    return CVX_SOLVED;
  }
  else if (status == GRB_INFEASIBLE)
  {
    GRBcomputeIIS(m_model);
    return CVX_INFEASIBLE;
  }
  else
    return CVX_FAILED;
}
CvxOptStatus GurobiModel::optimizeFeasRelax()
{
  double lbpen = GRB_INFINITY, ubpen = GRB_INFINITY, rhspen = 1;
  ENSURE_SUCCESS(
      GRBfeasrelax(m_model, 0 /*sum of viol*/, 0 /*just minimize cost of viol*/, &lbpen, &ubpen, &rhspen, nullptr));
  return optimize();
}

void GurobiModel::setObjective(const AffExpr& expr)
{
  GRBdelq(m_model);

  int nvars;
  GRBgetintattr(m_model, GRB_INT_ATTR_NUMVARS, &nvars);
  assert(nvars == static_cast<int>(m_vars.size()));

  DblVec obj(static_cast<std::size_t>(nvars), 0);
  for (std::size_t i = 0; i < expr.size(); ++i)
  {
    obj[expr.vars[i].var_rep->index] += expr.coeffs[i];
  }
  ENSURE_SUCCESS(GRBsetdblattrarray(m_model, "Obj", 0, nvars, obj.data()));
  GRBsetdblattr(m_model, "ObjCon", expr.constant);
}

void GurobiModel::setObjective(const QuadExpr& quad_expr)
{
  setObjective(quad_expr.affexpr);
  IntVec inds1;
  vars2inds(quad_expr.vars1, inds1);
  IntVec inds2;
  vars2inds(quad_expr.vars2, inds2);
  GRBaddqpterms(m_model,
                static_cast<int>(quad_expr.coeffs.size()),
                const_cast<int*>(inds1.data()),
                const_cast<int*>(inds2.data()),
                const_cast<double*>(quad_expr.coeffs.data()));
}

void GurobiModel::writeToFile(const std::string& fname) const { ENSURE_SUCCESS(GRBwrite(m_model, fname.c_str())); }
void GurobiModel::update()
{
  ENSURE_SUCCESS(GRBupdatemodel(m_model));

  {
    std::size_t inew = 0;
    for (Var& var : m_vars)
    {
      if (!var.var_rep->removed)
      {
        m_vars[inew] = var;
        var.var_rep->index = inew;
        ++inew;
      }
      else
      {
        var.var_rep = nullptr;
      }
    }
    m_vars.resize(inew);
  }
  {
    std::size_t inew = 0;
    for (Cnt& cnt : m_cnts)
    {
      if (!cnt.cnt_rep->removed)
      {
        m_cnts[inew] = cnt;
        cnt.cnt_rep->index = inew;
        ++inew;
      }
      else
      {
        cnt.cnt_rep = nullptr;
      }
    }
    m_cnts.resize(inew);
  }
}

VarVector GurobiModel::getVars() const { return m_vars; }
GurobiModel::~GurobiModel() { ENSURE_SUCCESS(GRBfreemodel(m_model)); }
}  // namespace sco
