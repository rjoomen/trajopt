#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Core>
#include <gtest/gtest.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <memory>
#include <optional>

#include <trajopt_sco/expr_op_overloads.hpp>
#include <trajopt_sco/modeling_utils.hpp>
#include <trajopt_sco/optimizers.hpp>
#include <trajopt_sco/sco_common.hpp>
#include <tesseract/common/logging.h>

using namespace sco;

namespace
{
double targetCost(const Eigen::VectorXd& x) { return sq(x(0) - 0.8) + sq(x(1) + 0.3); }

/** @brief Two variables in [-1, 1] pulled toward (0.8, -0.3); with a pin, x0 = pin is a hard nonlinear-API constraint
 */
OptProb::Ptr makeProblem(std::optional<double> pin = std::nullopt)
{
  auto prob = std::make_shared<OptProb>(ModelType::OSQP);
  prob->createVariables({ "x0", "x1" }, { -1.0, -1.0 }, { 1.0, 1.0 });
  prob->addCost(std::make_shared<CostFromFunc>(ScalarOfVector::construct(&targetCost), prob->getVars(), "target"));
  if (pin)
  {
    const double target = *pin;
    auto err = VectorOfVector::construct([target](const Eigen::VectorXd& x) {
      Eigen::VectorXd out(1);
      out(0) = x(0) - target;
      return out;
    });
    prob->addConstraint(std::make_shared<ConstraintFromErrFunc>(err, prob->getVars(), Eigen::VectorXd(), EQ, "pin"));
  }
  return prob;
}
}  // namespace

class ScoTermination : public testing::Test
{
protected:
  void SetUp() override { tesseract::common::getLogger()->set_level(spdlog::level::err); }
};

TEST_F(ScoTermination, IterationLimitKeepsItsStatusWhenFeasible)  // NOLINT
{
  BasicTrustRegionSQP solver(makeProblem());
  solver.getParameters().max_iter = 1;
  solver.initialize({ 0.0, 0.0 });
  EXPECT_EQ(solver.optimize(), OPT_SCO_ITERATION_LIMIT);
  EXPECT_TRUE(solver.results().best_is_feasible);
  EXPECT_TRUE(isUsable(solver.results()));
}

TEST_F(ScoTermination, TimeLimitBeforeAnySolveJudgesTheStartPoint)  // NOLINT
{
  BasicTrustRegionSQP solver(makeProblem(0.0));  // start point satisfies the pin
  solver.getParameters().max_time = -1.0;
  solver.initialize({ 0.0, 0.0 });
  EXPECT_EQ(solver.optimize(), OPT_TIME_LIMIT);
  EXPECT_EQ(solver.results().n_qp_solves, 0);
  EXPECT_TRUE(solver.results().best_is_feasible);
}

TEST_F(ScoTermination, TimeLimitWithViolatedStartIsNotUsable)  // NOLINT
{
  BasicTrustRegionSQP solver(makeProblem(0.5));
  solver.getParameters().max_time = -1.0;
  solver.initialize({ 0.0, 0.0 });
  EXPECT_EQ(solver.optimize(), OPT_TIME_LIMIT);
  EXPECT_FALSE(solver.results().best_is_feasible);
  EXPECT_FALSE(isUsable(solver.results()));
}

TEST(ScoIsUsable, EveryStatusAndFeasibility)  // NOLINT
{
  OptResults results;
  for (const bool feasible : { false, true })
  {
    results.best_is_feasible = feasible;
    for (const OptStatus status :
         { OPT_CONVERGED, OPT_SCO_ITERATION_LIMIT, OPT_PENALTY_ITERATION_LIMIT, OPT_TIME_LIMIT, OPT_FAILED })
    {
      results.status = status;
      EXPECT_EQ(isUsable(results), feasible) << toString(status);
    }
    results.status = INVALID;
    EXPECT_FALSE(isUsable(results));
  }
}
