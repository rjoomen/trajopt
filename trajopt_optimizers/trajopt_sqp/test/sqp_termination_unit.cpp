#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <tesseract/common/logging.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <memory>
#include <optional>

#include <trajopt_ifopt/constraints/joint_position_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <trajopt_sqp/osqp_eigen_solver.h>
#include <trajopt_sqp/trajopt_qp_problem.h>
#include <trajopt_sqp/trust_region_sqp_solver.h>
#include <trajopt_sqp/types.h>

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
    EXPECT_FALSE(trajopt_sqp::isUsable(SQPStatus::kRunning, results));
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
