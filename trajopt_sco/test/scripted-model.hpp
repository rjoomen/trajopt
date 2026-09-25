#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <trajopt_sco/modeling.hpp>
#include <trajopt_sco/solver_interface.hpp>

namespace sco::test
{
/** @brief One scripted override of an optimize(); an empty field leaves the inner model's result as it is */
struct ScriptedModelSolve
{
  std::optional<CvxOptStatus> status;
  /** @brief Edits the full model variable vector (problem variables first, then auxiliaries) */
  std::function<void(DblVec&)> edit_solution;
  std::optional<double> duality_gap;
};

/**
 * @brief Forwards to a real model, then applies the front of its script to each optimize() outcome
 * @details After an inner solve that fails, reports NaN for every variable, since the inner model may hold no values
 */
class ScriptedModel : public Model
{
public:
  explicit ScriptedModel(Model::Ptr inner) : inner_(std::move(inner)) {}

  /** @brief Consumed front first; once empty, solves pass through unchanged */
  std::deque<ScriptedModelSolve> script;
  /** @brief Applied to every solve after the script is exhausted, when set */
  std::optional<ScriptedModelSolve> every_solve;
  int solves{ 0 };

  Var addVar(const std::string& name) override { return inner_->addVar(name); }
  Cnt addEqCnt(const AffExpr& e, const std::string& name) override { return inner_->addEqCnt(e, name); }
  Cnt addIneqCnt(const AffExpr& e, const std::string& name) override { return inner_->addIneqCnt(e, name); }
  Cnt addIneqCnt(const QuadExpr& e, const std::string& name) override { return inner_->addIneqCnt(e, name); }
  void removeVars(const VarVector& vars) override { inner_->removeVars(vars); }
  void removeCnts(const CntVector& cnts) override { inner_->removeCnts(cnts); }
  void update() override { inner_->update(); }
  void setVarBounds(const VarVector& vars, const DblVec& lower, const DblVec& upper) override
  {
    inner_->setVarBounds(vars, lower, upper);
  }
  DblVec getVarValues(const VarVector& vars) const override
  {
    DblVec out(vars.size());
    for (std::size_t i = 0; i < vars.size(); ++i)
      out[i] = values_[vars[i].var_rep->index];
    return out;
  }
  CvxOptStatus optimize() override
  {
    ++solves;
    CvxOptStatus status = inner_->optimize();
    values_ = (status == CVX_SOLVED) ? inner_->getVarValues(inner_->getVars()) :
                                       DblVec(inner_->getVars().size(), std::numeric_limits<double>::quiet_NaN());
    gap_ = inner_->getDualityGap();
    std::optional<ScriptedModelSolve> step = every_solve;
    if (!script.empty())
    {
      step = script.front();
      script.pop_front();
    }
    if (step)
    {
      if (step->status)
        status = *step->status;
      if (step->edit_solution)
        step->edit_solution(values_);
      if (step->duality_gap)
        gap_ = *step->duality_gap;
    }
    return status;
  }
  double getDualityGap() const override { return gap_; }
  void setObjective(const AffExpr& e) override { inner_->setObjective(e); }
  void setObjective(const QuadExpr& e) override { inner_->setObjective(e); }
  void writeToFile(const std::string& fname) const override { inner_->writeToFile(fname); }
  VarVector getVars() const override { return inner_->getVars(); }

private:
  Model::Ptr inner_;
  DblVec values_;
  double gap_{ 0 };
};

/** @brief An OptProb whose convex model is the given one, so a test can script its solves */
class ScriptedProb : public OptProb
{
public:
  explicit ScriptedProb(Model::Ptr model) : OptProb(ModelType::OSQP) { model_ = std::move(model); }
};
}  // namespace sco::test
