/**
 * @file osqp_eigen_solver_unit.cpp
 * @brief Tests the status handling of OSQPEigenSolver
 *
 * @author Roelof Oomen
 * @date September 24, 2026
 *
 * @copyright Copyright (c) 2026, Roelof Oomen
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <algorithm>
#include <cmath>
#include <tesseract/common/logging.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
TRAJOPT_IGNORE_WARNINGS_POP

TRAJOPT_IGNORE_WARNINGS_PUSH
#include <OsqpEigen/OsqpEigen.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_ifopt/core/eigen_types.h>
#include <trajopt_sqp/osqp_eigen_solver.h>
#include "scripted_qp_solver.h"

using trajopt_sqp::OSQPEigenSolver;
using trajopt_sqp::QPSolverStatus;
using trajopt_sqp::test::setupOneSidedProblem;

TEST(OSQPEigenSolverUnit, StatusClassification)  // NOLINT
{
  using trajopt_sqp::QPSolveStatus;
  EXPECT_EQ(OSQPEigenSolver::toQPSolveStatus(OsqpEigen::Status::Solved), QPSolveStatus::kSolved);
  EXPECT_EQ(OSQPEigenSolver::toQPSolveStatus(OsqpEigen::Status::SolvedInaccurate), QPSolveStatus::kUnconverged);
  EXPECT_EQ(OSQPEigenSolver::toQPSolveStatus(OsqpEigen::Status::PrimalInfeasible), QPSolveStatus::kFailed);
  EXPECT_EQ(OSQPEigenSolver::toQPSolveStatus(OsqpEigen::Status::DualInfeasible), QPSolveStatus::kFailed);
  EXPECT_EQ(OSQPEigenSolver::toQPSolveStatus(OsqpEigen::Status::MaxIterReached), QPSolveStatus::kUnconverged);
}

TEST(OSQPEigenSolverUnit, SuccessfulSolveClearsFailure)  // NOLINT
{
  // Minimize x'x subject to x0 + x1 >= 3 and x0 + x1 <= 1: infeasible until the lower bound is dropped
  constexpr double inf = std::numeric_limits<double>::infinity();
  const std::vector<Eigen::Triplet<double>> triplets{ { 0, 0, 1.0 }, { 0, 1, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 } };
  trajopt_ifopt::Jacobian A(2, 2);
  A.setFromTriplets(triplets.begin(), triplets.end());
  trajopt_ifopt::Jacobian hessian(2, 2);
  hessian.setIdentity();

  OSQPEigenSolver solver;
  solver.init(2, 2);
  solver.updateHessianMatrix(hessian);
  solver.updateGradient(Eigen::Vector2d::Zero());
  solver.updateLinearConstraintsMatrix(A);
  solver.updateBounds(Eigen::Vector2d(3.0, -inf), Eigen::Vector2d(inf, 1.0));
  EXPECT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kFailed);
  EXPECT_EQ(solver.getSolverStatus(), QPSolverStatus::kFailed);

  solver.updateBounds(Eigen::Vector2d(-inf, -inf), Eigen::Vector2d(inf, 1.0));
  ASSERT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kSolved);
  EXPECT_EQ(solver.getSolverStatus(), QPSolverStatus::kInitialized);
}

TEST(OSQPEigenSolverUnit, DualityGapMatchesPrimalMinusDualObjective)  // NOLINT
{
  OSQPEigenSolver solver;
  setupOneSidedProblem(solver);
  ASSERT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kSolved);

  const double x = solver.getSolution()[0];
  const double y = solver.solver_->getDualSolution()[0];
  // OSQP's P is twice the QP Hessian: primal 0.5 x'Px + q'x, dual -0.5 x'Px - u * max(y, 0) with only u finite
  const double primal = (x * x) - (2.0 * x);
  const double dual = -(x * x) - (0.5 * std::max(y, 0.0));
  EXPECT_NEAR(solver.getDualityGap(), std::abs(primal - dual), 1e-6);
  EXPECT_LT(solver.getDualityGap(), 1e-4);
}

TEST(OSQPEigenSolverUnit, IterationCapReturnsAnUnconvergedSolutionWithFiniteGap)  // NOLINT
{
  // Silence the unconverged-solve WARN that every capped solve logs
  tesseract::common::getLogger()->set_level(spdlog::level::off);
  OSQPEigenSolver solver;
  solver.solver_->settings()->setMaxIteration(1);
  solver.solver_->settings()->setPolish(false);
  setupOneSidedProblem(solver);
  EXPECT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kUnconverged);
  EXPECT_EQ(solver.getSolution().size(), 1);
  EXPECT_TRUE(solver.getSolution().allFinite());
  EXPECT_TRUE(std::isfinite(solver.getDualityGap()));
}

TEST(OSQPEigenSolverUnit, DualityGapInfiniteBeforeAnySolve)  // NOLINT
{
  OSQPEigenSolver solver;
  setupOneSidedProblem(solver);
  EXPECT_EQ(solver.getDualityGap(), std::numeric_limits<double>::infinity());
}

TEST(OSQPEigenSolverUnit, DualityGapInfiniteAfterInfeasibleSolve)  // NOLINT
{
  // Minimize x'x subject to x0 + x1 >= 3 and x0 + x1 <= 1 on separate rows: each row is valid on its own, but
  // together infeasible, so OSQP detects it during the solve rather than rejecting the bounds update outright
  constexpr double inf = std::numeric_limits<double>::infinity();
  const std::vector<Eigen::Triplet<double>> triplets{ { 0, 0, 1.0 }, { 0, 1, 1.0 }, { 1, 0, 1.0 }, { 1, 1, 1.0 } };
  trajopt_ifopt::Jacobian A(2, 2);
  A.setFromTriplets(triplets.begin(), triplets.end());
  trajopt_ifopt::Jacobian hessian(2, 2);
  hessian.setIdentity();

  OSQPEigenSolver solver;
  solver.init(2, 2);
  solver.updateHessianMatrix(hessian);
  solver.updateGradient(Eigen::Vector2d::Zero());
  solver.updateLinearConstraintsMatrix(A);
  solver.updateBounds(Eigen::Vector2d(-inf, -inf), Eigen::Vector2d(inf, 1.0));
  ASSERT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kSolved);
  ASSERT_TRUE(std::isfinite(solver.getDualityGap()));

  // Reuse the same solver so a stale finite gap left over from the feasible solve would show
  solver.updateBounds(Eigen::Vector2d(3.0, -inf), Eigen::Vector2d(inf, 1.0));
  EXPECT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kFailed);
  EXPECT_EQ(solver.getDualityGap(), inf);
}

TEST(OSQPEigenSolverUnit, DualityGapInfiniteAfterClear)  // NOLINT
{
  OSQPEigenSolver solver;
  setupOneSidedProblem(solver);
  ASSERT_EQ(solver.solve(), trajopt_sqp::QPSolveStatus::kSolved);
  ASSERT_TRUE(std::isfinite(solver.getDualityGap()));

  ASSERT_TRUE(solver.clear());
  EXPECT_EQ(solver.getDualityGap(), std::numeric_limits<double>::infinity());
}

TEST(OSQPEigenSolverUnit, SmallGradientEntriesReachTheSolver)  // NOLINT
{
  // The solved QP must be the scored model: a 5e-8 gradient entry must not be dropped
  OSQPEigenSolver solver;
  trajopt_ifopt::Jacobian A(1, 1);
  A.insert(0, 0) = 1.0;
  trajopt_ifopt::Jacobian hessian(1, 1);
  hessian.insert(0, 0) = 1.0;
  solver.init(1, 1);
  solver.updateHessianMatrix(hessian);
  solver.updateGradient(Eigen::VectorXd::Constant(1, 5e-8));
  solver.updateLinearConstraintsMatrix(A);
  solver.updateBounds(Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0));
  EXPECT_DOUBLE_EQ(solver.solver_->data()->getData()->q[0], 5e-8);
}
