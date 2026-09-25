#ifndef TRAJOPT_SQP_TEST_SCRIPTED_QP_SOLVER_H
#define TRAJOPT_SQP_TEST_SCRIPTED_QP_SOLVER_H

#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <trajopt_sqp/qp_problem.h>
#include <trajopt_sqp/qp_solver.h>

namespace trajopt_sqp::test
{
/** @brief One scripted override of a solve; an empty field leaves the inner solver's result as it is */
struct ScriptedSolve
{
  std::optional<bool> succeeded;
  std::function<void(Eigen::VectorXd&)> edit_solution;
  std::optional<double> duality_gap;
};

/** @brief Forwards to a real solver, then applies the front of its script to each solve's outcome */
class ScriptedQPSolver : public QPSolver
{
public:
  explicit ScriptedQPSolver(std::shared_ptr<QPSolver> inner) : inner_(std::move(inner)) {}

  /** @brief Consumed front first; once empty, solves pass through unchanged */
  std::deque<ScriptedSolve> script;
  /** @brief Applied to every solve after the script is exhausted, when set */
  std::optional<ScriptedSolve> every_solve;
  int solves{ 0 };

  bool init(Eigen::Index num_vars, Eigen::Index num_cnts) override { return inner_->init(num_vars, num_cnts); }
  bool clear() override { return inner_->clear(); }
  bool solve() override
  {
    ++solves;
    bool succeeded = inner_->solve();
    solution_ = inner_->getSolution();
    gap_ = inner_->getDualityGap();
    std::optional<ScriptedSolve> step = every_solve;
    if (!script.empty())
    {
      step = script.front();
      script.pop_front();
    }
    if (step)
    {
      if (step->succeeded)
        succeeded = *step->succeeded;
      if (step->edit_solution)
        step->edit_solution(solution_);
      if (step->duality_gap)
        gap_ = *step->duality_gap;
    }
    return succeeded;
  }
  Eigen::VectorXd getSolution() override { return solution_; }
  double getDualityGap() const override { return gap_; }
  bool updateHessianMatrix(const trajopt_ifopt::Jacobian& hessian) override
  {
    return inner_->updateHessianMatrix(hessian);
  }
  bool updateGradient(const Eigen::Ref<const Eigen::VectorXd>& gradient) override
  {
    return inner_->updateGradient(gradient);
  }
  bool updateLowerBound(const Eigen::Ref<const Eigen::VectorXd>& lower) override
  {
    return inner_->updateLowerBound(lower);
  }
  bool updateUpperBound(const Eigen::Ref<const Eigen::VectorXd>& upper) override
  {
    return inner_->updateUpperBound(upper);
  }
  bool updateBounds(const Eigen::Ref<const Eigen::VectorXd>& lower,
                    const Eigen::Ref<const Eigen::VectorXd>& upper) override
  {
    return inner_->updateBounds(lower, upper);
  }
  bool updateLinearConstraintsMatrix(const trajopt_ifopt::Jacobian& matrix) override
  {
    return inner_->updateLinearConstraintsMatrix(matrix);
  }
  bool setWarmStart(const QPProblem& qp_problem) override { return inner_->setWarmStart(qp_problem); }
  QPSolverStatus getSolverStatus() const override { return inner_->getSolverStatus(); }

private:
  std::shared_ptr<QPSolver> inner_;
  Eigen::VectorXd solution_;
  double gap_{ 0 };
};

/**
 * @brief Forwards to a real problem; can replace exact costs and exact constraint violations by call index, and
 * records non-finite setVariables
 */
class ScriptedQPProblem : public QPProblem
{
public:
  explicit ScriptedQPProblem(std::shared_ptr<QPProblem> inner) : inner_(std::move(inner)) {}

  /** @brief Called with the 1-based exact cost call index and the inner costs; returns the costs to report */
  std::function<Eigen::VectorXd(int, const Eigen::VectorXd&)> exact_costs_hook;
  /** @brief Counts exact cost evaluations: every getExactCosts call, including the one inside getTotalExactCost */
  mutable int exact_cost_calls{ 0 };
  /**
   * @brief Called with the 1-based getExactConstraintViolations call index and the inner violations; returns the
   * violations to report
   */
  std::function<ConstraintViolations(int, const ConstraintViolations&)> exact_violations_hook;
  /** @brief Counts getExactConstraintViolations calls */
  mutable int exact_violation_calls{ 0 };
  bool saw_non_finite_variables{ false };

  void addConstraintSet(std::shared_ptr<trajopt_ifopt::ConstraintSet> c) override { inner_->addConstraintSet(c); }
  void addCostSet(std::shared_ptr<trajopt_ifopt::ConstraintSet> c, CostPenaltyType t) override
  {
    inner_->addCostSet(c, t);
  }
  void setup() override { inner_->setup(); }
  void setVariables(const double* x) override
  {
    for (Eigen::Index i = 0; i < inner_->getNumNLPVars(); ++i)
      saw_non_finite_variables |= !std::isfinite(x[i]);
    inner_->setVariables(x);
  }
  Eigen::VectorXd getVariableValues() const override { return inner_->getVariableValues(); }
  void convexify() override { inner_->convexify(); }
  double evaluateTotalConvexCost(const Eigen::Ref<const Eigen::VectorXd>& v) const override
  {
    return inner_->evaluateTotalConvexCost(v);
  }
  Eigen::VectorXd evaluateConvexCosts(const Eigen::Ref<const Eigen::VectorXd>& v) const override
  {
    return inner_->evaluateConvexCosts(v);
  }
  double getTotalExactCost() const override { return getExactCosts().sum(); }
  Eigen::VectorXd getExactCosts() const override
  {
    ++exact_cost_calls;
    Eigen::VectorXd costs = inner_->getExactCosts();
    return exact_costs_hook ? exact_costs_hook(exact_cost_calls, costs) : costs;
  }
  ConstraintViolations evaluateConvexConstraintViolations(const Eigen::Ref<const Eigen::VectorXd>& v) const override
  {
    return inner_->evaluateConvexConstraintViolations(v);
  }
  ConstraintViolations getExactConstraintViolations() const override
  {
    ++exact_violation_calls;
    ConstraintViolations violations = inner_->getExactConstraintViolations();
    return exact_violations_hook ? exact_violations_hook(exact_violation_calls, violations) : violations;
  }
  void scaleBoxSize(double& scale) override { inner_->scaleBoxSize(scale); }
  void setBoxSize(const Eigen::Ref<const Eigen::VectorXd>& b) override { inner_->setBoxSize(b); }
  void setConstraintMeritCoeff(const Eigen::Ref<const Eigen::VectorXd>& c) override
  {
    inner_->setConstraintMeritCoeff(c);
  }
  void print() const override { inner_->print(); }
  Eigen::Index getNumNLPVars() const override { return inner_->getNumNLPVars(); }
  Eigen::Index getNumNLPConstraints() const override { return inner_->getNumNLPConstraints(); }
  Eigen::Index getNumNLPCosts() const override { return inner_->getNumNLPCosts(); }
  Eigen::Index getNumQPVars() const override { return inner_->getNumQPVars(); }
  Eigen::Index getNumQPConstraints() const override { return inner_->getNumQPConstraints(); }
  const std::vector<std::string>& getNLPConstraintNames() const override { return inner_->getNLPConstraintNames(); }
  const std::vector<std::string>& getNLPCostNames() const override { return inner_->getNLPCostNames(); }
  const Eigen::VectorXd& getBoxSize() const override { return inner_->getBoxSize(); }
  const Eigen::VectorXd& getConstraintMeritCoeff() const override { return inner_->getConstraintMeritCoeff(); }
  const trajopt_ifopt::Jacobian& getHessian() const override { return inner_->getHessian(); }
  const Eigen::VectorXd& getGradient() const override { return inner_->getGradient(); }
  const trajopt_ifopt::Jacobian& getConstraintMatrix() const override { return inner_->getConstraintMatrix(); }
  const Eigen::VectorXd& getBoundsLower() const override { return inner_->getBoundsLower(); }
  const Eigen::VectorXd& getBoundsUpper() const override { return inner_->getBoundsUpper(); }

private:
  std::shared_ptr<QPProblem> inner_;
};
}  // namespace trajopt_sqp::test

#endif
