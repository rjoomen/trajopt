#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <tesseract/common/logging.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <cmath>
#include <limits>
#include <memory>
#include <optional>

#include <trajopt_ifopt/constraints/joint_position_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <trajopt_sqp/osqp_eigen_solver.h>
#include <trajopt_sqp/sqp_callback.h>
#include <trajopt_sqp/trajopt_qp_problem.h>
#include <trajopt_sqp/trust_region_sqp_solver.h>
#include <trajopt_sqp/types.h>

#include "scripted_qp_solver.h"

using trajopt_sqp::SQPStatus;

namespace
{
/**
 * @brief Two variables in [-1, 1] pulled toward (0.8, -0.3) by a squared cost, optionally constrained to x0 = target
 * @param constraint_target When set, adds the hard constraint x0 = constraint_target (outside [-1, 1] is infeasible)
 */
std::shared_ptr<trajopt_sqp::TrajOptQPProblem> makeProblem(std::optional<double> constraint_target = std::nullopt)
{
  auto node = std::make_unique<trajopt_ifopt::Node>("Joints");
  const std::vector<std::string> names{ "j0", "j1" };
  const std::vector<trajopt_ifopt::Bounds> bounds(2, trajopt_ifopt::Bounds(-1.0, 1.0));
  const std::shared_ptr<const trajopt_ifopt::Var> var =
      node->addVar("position", names, Eigen::Vector2d::Zero(), bounds);
  std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
  nodes.push_back(std::move(node));
  auto variables = std::make_shared<trajopt_ifopt::NodesVariables>("trajectory", std::move(nodes));

  auto qp = std::make_shared<trajopt_sqp::TrajOptQPProblem>(variables);
  auto cost = std::make_shared<trajopt_ifopt::JointPosConstraint>(
      Eigen::Vector2d(0.8, -0.3), var, Eigen::VectorXd::Ones(1), "Target");
  qp->addCostSet(cost, trajopt_sqp::CostPenaltyType::kSquared);
  if (constraint_target)
  {
    // Pin only x0.
    const std::vector<trajopt_ifopt::Bounds> cnt_bounds{ trajopt_ifopt::Bounds(*constraint_target,
                                                                               *constraint_target) };
    auto cnt = std::make_shared<trajopt_ifopt::JointPosConstraint>(cnt_bounds, var, Eigen::VectorXd::Ones(1), "Pin");
    qp->addConstraintSet(cnt);
  }
  qp->setup();
  return qp;
}

trajopt_sqp::TrustRegionSQPSolver makeSolver(std::shared_ptr<trajopt_sqp::QPSolver> qp_solver = nullptr)
{
  if (!qp_solver)
    qp_solver = std::make_shared<trajopt_sqp::OSQPEigenSolver>();
  return { std::move(qp_solver) };
}
}  // namespace

class SQPTermination : public testing::Test
{
protected:
  void SetUp() override { tesseract::common::getLogger()->set_level(spdlog::level::off); }
};

TEST_F(SQPTermination, IterationLimitKeepsItsStatusWhenFeasible)  // NOLINT
{
  auto solver = makeSolver();
  solver.params.max_iterations = 1;
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kIterationLimit);
  EXPECT_TRUE(solver.getResults().best_is_feasible);
  EXPECT_TRUE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

TEST_F(SQPTermination, IterationLimitReportsInfeasibleBest)  // NOLINT
{
  auto solver = makeSolver();
  solver.params.max_iterations = 1;
  solver.solve(makeProblem(5.0));
  EXPECT_EQ(solver.getStatus(), SQPStatus::kIterationLimit);
  EXPECT_FALSE(solver.getResults().best_is_feasible);
  EXPECT_FALSE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

TEST_F(SQPTermination, TimeLimitBeforeAnySolveJudgesTheStartPoint)  // NOLINT
{
  auto solver = makeSolver();
  solver.params.max_time = -1.0;
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kTimeLimit);
  EXPECT_EQ(solver.getResults().overall_iteration, 0);
  EXPECT_TRUE(solver.getResults().best_is_feasible);
}

TEST_F(SQPTermination, TimeLimitWithViolatedStartIsNotUsable)  // NOLINT
{
  auto solver = makeSolver();
  solver.params.max_time = -1.0;
  solver.solve(makeProblem(5.0));
  EXPECT_EQ(solver.getStatus(), SQPStatus::kTimeLimit);
  EXPECT_FALSE(solver.getResults().best_is_feasible);
  EXPECT_FALSE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

TEST_F(SQPTermination, ConvergedRunIsFeasibleAndUsable)  // NOLINT
{
  auto solver = makeSolver();
  solver.solve(makeProblem(0.5));
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
  EXPECT_TRUE(solver.getResults().best_is_feasible);
}

TEST(SQPIsUsable, EveryStatusAndFeasibility)  // NOLINT
{
  trajopt_sqp::SQPResults results;
  for (const bool feasible : { false, true })
  {
    results.best_is_feasible = feasible;
    EXPECT_EQ(trajopt_sqp::isUsable(SQPStatus::kConverged, results), feasible);
    EXPECT_EQ(trajopt_sqp::isUsable(SQPStatus::kIterationLimit, results), feasible);
    EXPECT_EQ(trajopt_sqp::isUsable(SQPStatus::kPenaltyIterationLimit, results), feasible);
    EXPECT_EQ(trajopt_sqp::isUsable(SQPStatus::kTimeLimit, results), feasible);
    EXPECT_EQ(trajopt_sqp::isUsable(SQPStatus::kQPSolveFailed, results), feasible);
    EXPECT_FALSE(trajopt_sqp::isUsable(SQPStatus::kStoppedByCallback, results));
    EXPECT_FALSE(trajopt_sqp::isUsable(SQPStatus::kNonFiniteMerit, results));
    EXPECT_FALSE(trajopt_sqp::isUsable(SQPStatus::kRunning, results));
  }
}

namespace
{
/** @brief Counts its calls and returns a fixed verdict */
class CountingCallback : public trajopt_sqp::SQPCallback
{
public:
  explicit CountingCallback(bool verdict) : verdict_(verdict) {}
  bool execute(const trajopt_sqp::QPProblem& /*problem*/, const trajopt_sqp::SQPResults& /*results*/) override
  {
    ++calls;
    return verdict_;
  }
  int calls{ 0 };

private:
  bool verdict_;
};
}  // namespace

TEST_F(SQPTermination, SpentFailureBudgetEndsTheSolve)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->every_solve = trajopt_sqp::test::ScriptedSolve{ false, nullptr, std::nullopt };
  auto solver = makeSolver(scripted);
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kQPSolveFailed);
  EXPECT_EQ(scripted->solves, solver.params.max_qp_solver_failures + 1);
  EXPECT_EQ(solver.getResults().overall_iteration, solver.params.max_qp_solver_failures + 1);
}

TEST_F(SQPTermination, FailureShrinkingToTinyBoxIsATinyBoxExit)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->script.push_back({ false, nullptr, std::nullopt });
  auto solver = makeSolver(scripted);
  solver.params.initial_trust_box_size = 1.5e-4;  // one shrink by trust_shrink_ratio lands below min_trust_box_size
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
  EXPECT_EQ(scripted->solves, 1);
}

TEST_F(SQPTermination, CallbackStopEndsTheSolve)  // NOLINT
{
  auto solver = makeSolver();
  auto callback = std::make_shared<CountingCallback>(false);
  solver.registerCallback(callback);
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kStoppedByCallback);
  EXPECT_EQ(callback->calls, 1);
  EXPECT_FALSE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

namespace
{
/** @brief Records whether any solve saw a non-finite best merit; never stops the solve */
class BestMeritWatcher : public trajopt_sqp::SQPCallback
{
public:
  bool execute(const trajopt_sqp::QPProblem& /*problem*/, const trajopt_sqp::SQPResults& results) override
  {
    saw_non_finite_best |= !std::isfinite(results.best_exact_merit);
    return true;
  }
  bool saw_non_finite_best{ false };
};
}  // namespace

TEST_F(SQPTermination, NonFiniteSolutionIsAFailedSolveAndNeverEvaluated)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->script.push_back({ std::nullopt, [](Eigen::VectorXd& x) { x[0] = std::nan(""); }, std::nullopt });
  auto problem = std::make_shared<trajopt_sqp::test::ScriptedQPProblem>(makeProblem());
  auto solver = makeSolver(scripted);
  solver.solve(problem);
  EXPECT_FALSE(problem->saw_non_finite_variables);
  EXPECT_GT(scripted->solves, 1);
  EXPECT_TRUE(solver.getResults().best_var_vals.allFinite());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
}

TEST_F(SQPTermination, NonFiniteTrialMeritIsRejected)  // NOLINT
{
  auto problem = std::make_shared<trajopt_sqp::test::ScriptedQPProblem>(makeProblem());
  // Call 1 is the start point; call 2 is the first trial point
  problem->exact_costs_hook = [](int call, const Eigen::VectorXd& costs) {
    return call == 2 ? Eigen::VectorXd::Constant(costs.size(), std::nan("")) : costs;
  };
  auto solver = makeSolver();
  auto watcher = std::make_shared<BestMeritWatcher>();
  solver.registerCallback(watcher);
  solver.solve(problem);
  EXPECT_FALSE(watcher->saw_non_finite_best);
  EXPECT_TRUE(std::isfinite(solver.getResults().best_exact_merit));
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
  EXPECT_NEAR(solver.getResults().best_var_vals[0], 0.8, 1e-3);
}

TEST_F(SQPTermination, NonFiniteStartFailsFast)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  auto problem = std::make_shared<trajopt_sqp::test::ScriptedQPProblem>(makeProblem());
  problem->exact_costs_hook = [](int call, const Eigen::VectorXd& costs) {
    return call == 1 ? Eigen::VectorXd::Constant(costs.size(), std::numeric_limits<double>::infinity()) : costs;
  };
  auto solver = makeSolver(scripted);
  solver.solve(problem);
  EXPECT_EQ(solver.getStatus(), SQPStatus::kNonFiniteMerit);
  EXPECT_EQ(scripted->solves, 0);
  EXPECT_FALSE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

TEST_F(SQPTermination, InfiniteStartViolationFailsFast)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  auto problem = std::make_shared<trajopt_sqp::test::ScriptedQPProblem>(makeProblem(0.5));
  problem->exact_violations_hook = [](int call, const trajopt_sqp::ConstraintViolations& v) {
    if (call != 1)
      return v;
    trajopt_sqp::ConstraintViolations out = v;
    out.raw.setConstant(std::numeric_limits<double>::infinity());
    out.weighted.setConstant(std::numeric_limits<double>::infinity());
    return out;
  };
  auto solver = makeSolver(scripted);
  solver.solve(problem);
  EXPECT_EQ(solver.getStatus(), SQPStatus::kNonFiniteMerit);
  EXPECT_EQ(scripted->solves, 0);
  EXPECT_FALSE(trajopt_sqp::isUsable(solver.getStatus(), solver.getResults()));
}

TEST_F(SQPTermination, MoreThanNinetyNineConvexifyRoundsRun)  // NOLINT
{
  // A far target, a small fixed box and no expansion: every round accepts one short step
  auto node = std::make_unique<trajopt_ifopt::Node>("Joints");
  const std::shared_ptr<const trajopt_ifopt::Var> var =
      node->addVar("position", { "j0" }, Eigen::VectorXd::Zero(1), { trajopt_ifopt::Bounds(-1000.0, 1000.0) });
  std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
  nodes.push_back(std::move(node));
  auto qp = std::make_shared<trajopt_sqp::TrajOptQPProblem>(
      std::make_shared<trajopt_ifopt::NodesVariables>("trajectory", std::move(nodes)));
  qp->addCostSet(std::make_shared<trajopt_ifopt::JointPosConstraint>(
                     Eigen::VectorXd::Constant(1, 100.0), var, Eigen::VectorXd::Ones(1), "Far"),
                 trajopt_sqp::CostPenaltyType::kSquared);
  qp->setup();

  auto solver = makeSolver();
  solver.params.initial_trust_box_size = 0.5;
  solver.params.trust_expand_ratio = 1.0;
  solver.params.max_iterations = 150;
  solver.solve(qp);
  EXPECT_GT(solver.getResults().overall_iteration, 99);
  EXPECT_EQ(solver.getStatus(), SQPStatus::kIterationLimit);
}

// OSQP's own gap at a polished solution is far below min_approx_improve, so the certified test takes today's exit.
// Forcing the gap to exactly 0 here would misread OSQP's own round-off as a contradiction; a solver that reports 0
// solves exactly.
TEST_F(SQPTermination, CertifiedSolvesExitAsBefore)  // NOLINT
{
  auto solver = makeSolver();
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
  EXPECT_EQ(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kSmallImprovement);
  EXPECT_EQ(solver.getResults().n_suppressed_exits, 0);
}

TEST_F(SQPTermination, LargeGapSuppressesTheExit)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->every_solve = trajopt_sqp::test::ScriptedSolve{ std::nullopt, nullptr, 1.0 };
  auto solver = makeSolver(scripted);
  solver.solve(makeProblem());
  EXPECT_NE(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kSmallImprovement);
  EXPECT_GT(solver.getResults().n_suppressed_exits, 0);
}

TEST_F(SQPTermination, NaNGapIsUncertified)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->every_solve = trajopt_sqp::test::ScriptedSolve{ std::nullopt, nullptr, std::nan("") };
  auto solver = makeSolver(scripted);
  solver.solve(makeProblem());
  EXPECT_NE(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kSmallImprovement);
  EXPECT_GT(solver.getResults().n_suppressed_exits, 0);
}

TEST_F(SQPTermination, PredictionBelowMinusGapGoesToTheRatioTest)  // NOLINT
{
  // The first solution is moved away from the target, so the model predicts a merit increase
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->script.push_back({ std::nullopt, [](Eigen::VectorXd& x) { x[0] = -0.05; }, 0.0 });
  auto solver = makeSolver(scripted);
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getStatus(), SQPStatus::kConverged);
  EXPECT_NEAR(solver.getResults().best_var_vals[0], 0.8, 1e-3);
  EXPECT_GE(solver.getResults().n_suppressed_exits, 1);
}

TEST_F(SQPTermination, RatioExitRecordsItsReason)  // NOLINT
{
  auto solver = makeSolver();
  solver.params.min_approx_improve = 1e-12;
  solver.params.min_approx_improve_frac = 0.5;
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kSmallImprovementRatio);
}

TEST_F(SQPTermination, TinyBoxAfterUncertifiedRejectionsIsFlagged)  // NOLINT
{
  // Every trial point is charged +1 exact cost, so every step is rejected until the box is tiny
  auto make = [](double gap, bool& flag) {
    auto scripted =
        std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
    scripted->every_solve = trajopt_sqp::test::ScriptedSolve{ std::nullopt, nullptr, gap };
    auto problem = std::make_shared<trajopt_sqp::test::ScriptedQPProblem>(makeProblem());
    problem->exact_costs_hook = [](int call, const Eigen::VectorXd& costs) {
      return call == 1 ? costs : Eigen::VectorXd(costs.array() + 1.0);
    };
    auto solver = makeSolver(scripted);
    solver.params.min_approx_improve = 1e-12;  // reach the tiny box before a small-improvement exit
    solver.solve(problem);
    EXPECT_EQ(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kTinyTrustRegion);
    flag = solver.getResults().tiny_trust_region_after_uncertified;
  };
  bool certified_flag = true;
  bool uncertified_flag = false;
  make(0.0, certified_flag);
  make(1.0, uncertified_flag);
  EXPECT_FALSE(certified_flag);
  EXPECT_TRUE(uncertified_flag);
}

TEST_F(SQPTermination, TinyBoxAfterFailureShrinkIsFlagged)  // NOLINT
{
  auto scripted =
      std::make_shared<trajopt_sqp::test::ScriptedQPSolver>(std::make_shared<trajopt_sqp::OSQPEigenSolver>());
  scripted->script.push_back({ false, nullptr, std::nullopt });
  auto solver = makeSolver(scripted);
  solver.params.initial_trust_box_size = 1.5e-4;
  solver.solve(makeProblem());
  EXPECT_EQ(solver.getResults().exit_reason, trajopt_sqp::SQPExitReason::kTinyTrustRegion);
  EXPECT_TRUE(solver.getResults().tiny_trust_region_after_uncertified);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
