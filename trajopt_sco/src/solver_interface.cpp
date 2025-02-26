#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <boost/format.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <json/json.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_sco/solver_interface.hpp>

namespace sco
{
const std::vector<std::string> ModelType::MODEL_NAMES_ = { "GUROBI", "BPMPD", "OSQP", "QPOASES", "AUTO_SOLVER" };

void vars2inds(const VarVector& vars, SizeTVec& inds)
{
  inds = SizeTVec(vars.size());
  for (std::size_t i = 0; i < inds.size(); ++i)
    inds[i] = vars[i].var_rep->index;
}

void vars2inds(const VarVector& vars, IntVec& inds)
{
  inds = IntVec(vars.size());
  for (std::size_t i = 0; i < inds.size(); ++i)
    inds[i] = static_cast<int>(vars[i].var_rep->index);
}

void cnts2inds(const CntVector& cnts, SizeTVec& inds)
{
  inds = SizeTVec(cnts.size());
  for (std::size_t i = 0; i < inds.size(); ++i)
    inds[i] = cnts[i].cnt_rep->index;
}

void cnts2inds(const CntVector& cnts, IntVec& inds)
{
  inds = IntVec(cnts.size());
  for (std::size_t i = 0; i < inds.size(); ++i)
    inds[i] = static_cast<int>(cnts[i].cnt_rep->index);
}

void simplify2(IntVec& inds, DblVec& vals)
{
  using Int2Double = std::map<int, double>;
  Int2Double ind2val;
  for (unsigned i = 0; i < inds.size(); ++i)
  {
    if (vals[i] != 0.0)
      ind2val[inds[i]] += vals[i];
  }
  inds.resize(ind2val.size());
  vals.resize(ind2val.size());
  long unsigned int i_new = 0;
  for (const Int2Double::value_type& iv : ind2val)
  {
    inds[i_new] = iv.first;
    vals[i_new] = iv.second;
    ++i_new;
  }
}

AffExpr::AffExpr(double a) : constant(a) {}
AffExpr::AffExpr(const Var& v) : coeffs(1, 1), vars(1, v) {}
size_t AffExpr::size() const { return coeffs.size(); }

double AffExpr::value(const double* x) const
{
  double out = constant;
  for (std::size_t i = 0; i < size(); ++i)
  {
    out += coeffs[i] * vars[i].value(x);
  }
  return out;
}
double AffExpr::value(const DblVec& x) const
{
  double out = constant;
  for (std::size_t i = 0; i < size(); ++i)
  {
    out += coeffs[i] * vars[i].value(x);
  }
  return out;
}

QuadExpr::QuadExpr(double a) : affexpr(a) {}
QuadExpr::QuadExpr(const Var& v) : affexpr(v) {}
QuadExpr::QuadExpr(AffExpr aff) : affexpr(std::move(aff)) {}
size_t QuadExpr::size() const { return coeffs.size(); }

double QuadExpr::value(const DblVec& x) const
{
  double out = affexpr.value(x);
  for (std::size_t i = 0; i < size(); ++i)
  {
    out += coeffs[i] * vars1[i].value(x) * vars2[i].value(x);
  }
  return out;
}
double QuadExpr::value(const double* x) const
{
  double out = affexpr.value(x);
  for (std::size_t i = 0; i < size(); ++i)
  {
    out += coeffs[i] * vars1[i].value(x) * vars2[i].value(x);
  }
  return out;
}

Var Model::addVar(const std::string& name, double lb, double ub)
{
  Var v = addVar(name);
  setVarBounds(v, lb, ub);
  return v;
}
void Model::removeVar(const Var& var)
{
  const VarVector vars(1, var);
  removeVars(vars);
}
void Model::removeCnt(const Cnt& cnt)
{
  const CntVector cnts(1, cnt);
  removeCnts(cnts);
}

double Model::getVarValue(const Var& var) const
{
  const VarVector vars(1, var);
  return getVarValues(vars)[0];
}

void Model::setVarBounds(const Var& var, double lower, double upper)
{
  const DblVec lowers(1, lower);
  const DblVec uppers(1, upper);
  const VarVector vars(1, var);
  setVarBounds(vars, lowers, uppers);
}

std::ostream& operator<<(std::ostream& o, const Var& v)
{
  if (v.var_rep != nullptr)
    o << v.var_rep->name;
  else
    o << "nullvar";
  return o;
}

std::ostream& operator<<(std::ostream& o, const Cnt& c)
{
  o << c.cnt_rep->expr << ((c.cnt_rep->type == EQ) ? " == 0" : " <= 0");
  return o;
}

std::ostream& operator<<(std::ostream& o, const AffExpr& e)
{
  std::string sep;
  if (e.constant != 0)
  {
    o << e.constant;
    sep = " + ";
  }

  for (std::size_t i = 0; i < e.size(); ++i)
  {
    if (e.coeffs[i] != 0)
    {
      if (e.coeffs[i] == 1)
      {
        o << sep << e.vars[i];
      }
      else
      {
        o << sep << e.coeffs[i] << " " << e.vars[i];
      }
      sep = " + ";
    }
  }
  return o;
}

std::ostream& operator<<(std::ostream& o, const QuadExpr& e)
{
  o << e.affexpr;
  o << " + [ ";

  std::string op;
  for (std::size_t i = 0; i < e.size(); ++i)
  {
    if (e.coeffs[i] != 0)
    {
      o << op;
      if (e.coeffs[i] != 1)
      {
        o << e.coeffs[i] << " ";
      }
      if (e.vars1[i].var_rep->name == e.vars2[i].var_rep->name)
      {
        o << e.vars1[i] << " ^ 2";
      }
      else
      {
        o << e.vars1[i] << " * " << e.vars2[i];
      }
      op = " + ";
    }
  }
  o << " ] /2\n";
  return o;
}

std::ostream& operator<<(std::ostream& os, const ModelType& cs)
{
  auto cs_ivalue_ = static_cast<std::size_t>(cs.value_);
  if (cs_ivalue_ > ModelType::MODEL_NAMES_.size())
  {
    std::stringstream conversion_error;
    conversion_error << "Error converting ModelType to string - "
                     << "enum value is " << cs_ivalue_ << '\n';
    throw std::runtime_error(conversion_error.str());
  }
  os << ModelType::MODEL_NAMES_[cs_ivalue_];
  return os;
}

ModelType::ModelType() = default;
ModelType::ModelType(const ModelType::Value& v) : value_(v) {}
ModelType::ModelType(const int& v) : value_(static_cast<Value>(v)) {}
ModelType::ModelType(const std::string& s)
{
  for (unsigned int i = 0; i < ModelType::MODEL_NAMES_.size(); ++i)
  {
    if (s == ModelType::MODEL_NAMES_[i])
    {
      value_ = static_cast<ModelType::Value>(i);
      return;
    }
  }
  PRINT_AND_THROW(boost::format("invalid solver name:\"%s\"") % s);
}

ModelType::operator int() const { return static_cast<int>(value_); }
bool ModelType::operator==(const ModelType::Value& a) const { return value_ == a; }
bool ModelType::operator==(const ModelType& a) const { return value_ == a.value_; }
bool ModelType::operator!=(const ModelType& a) const { return value_ != a.value_; }
void ModelType::fromJson(const Json::Value& v)
{
  try
  {
    const std::string ref = v.asString();
    const ModelType cs(ref);
    value_ = cs.value_;
  }
  catch (const std::runtime_error&)
  {
    PRINT_AND_THROW(boost::format("expected: %s, got %s") % ("string") % (v));
  }
}

std::vector<ModelType> availableSolvers()
{
  std::vector<bool> has_solver(ModelType::AUTO_SOLVER, false);
#ifdef HAVE_GUROBI
  has_solver[ModelType::GUROBI] = true;
#endif
#ifdef HAVE_OSQP
  has_solver[ModelType::OSQP] = true;
#endif
#ifdef HAVE_BPMPD
  has_solver[ModelType::BPMPD] = true;
#endif
#ifdef HAVE_QPOASES
  has_solver[ModelType::QPOASES] = true;
#endif
  std::size_t n_available_solvers = 0;
  for (auto i = 0; i < ModelType::AUTO_SOLVER; ++i)
    if (has_solver[static_cast<std::size_t>(i)])
      ++n_available_solvers;
  std::vector<ModelType> available_solvers(n_available_solvers, ModelType::AUTO_SOLVER);

  std::size_t j = 0;
  for (int i = 0; i < static_cast<int>(ModelType::AUTO_SOLVER); ++i)
    if (has_solver[static_cast<std::size_t>(i)])
      available_solvers[j++] = static_cast<ModelType>(i);
  return available_solvers;
}

Model::Ptr createModel(ModelType model_type, const ModelConfig::ConstPtr& model_config)
{
  UNUSED(model_config);
#ifdef HAVE_GUROBI
  extern Model::Ptr createGurobiModel();
#endif
#ifdef HAVE_OSQP
  extern Model::Ptr createOSQPModel(const ModelConfig::ConstPtr& config);
#endif
#ifdef HAVE_BPMPD
  extern Model::Ptr createBPMPDModel();
#endif
#ifdef HAVE_QPOASES
  extern Model::Ptr createqpOASESModel();
#endif

  const char* solver_env = getenv("TRAJOPT_CONVEX_SOLVER");

  ModelType solver = model_type;

  if (solver == ModelType::AUTO_SOLVER)
  {
    if (solver_env != nullptr && std::string(solver_env) != "AUTO_SOLVER")
    {
      try
      {
        solver = ModelType(std::string(solver_env));
      }
      catch (std::runtime_error&)
      {
        PRINT_AND_THROW(boost::format("invalid solver \"%s\"specified by TRAJOPT_CONVEX_SOLVER") % solver_env);
      }
    }
    else
    {
      solver = availableSolvers()[0];
    }
  }

#ifndef HAVE_GUROBI
  if (solver == ModelType::GUROBI)
    PRINT_AND_THROW("you didn't build with GUROBI support");
#endif
#ifndef HAVE_OSQP
  if (solver == ModelType::OSQP)
    PRINT_AND_THROW("you don't have OSQP support on this platform");
#endif
#ifndef HAVE_BPMPD
  if (solver == ModelType::BPMPD)
    PRINT_AND_THROW("you don't have BPMPD support on this platform");
#endif
#ifndef HAVE_QPOASES
  if (solver == ModelType::QPOASES)
    PRINT_AND_THROW("you don't have qpOASES support on this platform");
#endif

#ifdef HAVE_GUROBI
  if (solver == ModelType::GUROBI)
    return createGurobiModel();
#endif
#ifdef HAVE_OSQP
  if (solver == ModelType::OSQP)
    return createOSQPModel(model_config);
#endif
#ifdef HAVE_BPMPD
  if (solver == ModelType::BPMPD)
    return createBPMPDModel();
#endif
#ifdef HAVE_QPOASES
  if (solver == ModelType::QPOASES)
    return createqpOASESModel();
#endif
  std::stringstream solver_instatiation_error;
  solver_instatiation_error << "Failed to create solver: unknown solver " << solver << '\n';
  PRINT_AND_THROW(solver_instatiation_error.str());
  return {};
}
}  // namespace sco
