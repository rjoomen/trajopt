/**
 * @file trust_region_sqp_solver.cpp
 * @brief Contains the main trust region SQP solver. While it is based on the paper below, it has been completely
 * rewritten from trajopt_sco
 *
 * Schulman, J., Ho, J., Lee, A. X., Awwal, I., Bradlow, H., & Abbeel, P. (2013, June). Finding Locally Optimal,
 * Collision-Free Trajectories with Sequential Convex Optimization. In Robotics: science and systems (Vol. 9, No. 1, pp.
 * 1-10).
 *
 * @author Matthew Powelson
 * @date May 18, 2020
 *
 * @copyright Copyright (c) 2020, Southwest Research Institute
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
#include <trajopt_ifopt/core/eigen_types.h>
#include <trajopt_sqp/trust_region_sqp_solver.h>
#include <trajopt_sqp/qp_problem.h>
#include <trajopt_sqp/qp_solver.h>
#include <trajopt_sqp/sqp_callback.h>

#include <tesseract/common/logging.h>
#include <chrono>
#include <cassert>
#include <cmath>
#include <limits>

namespace trajopt_sqp
{
namespace
{
/** @brief Evaluate the merit: the summed costs plus each merit unit's weighted violation times its coefficient. */
double meritValue(const Eigen::VectorXd& costs,
                  const ConstraintViolations& violations,
                  const Eigen::VectorXd& merit_error_coeffs)
{
  return costs.sum() + violations.weighted.dot(merit_error_coeffs);
}

/** @brief Whether a limit, a spent QP failure budget or a callback stop ends the solve with its own status */
bool isTerminal(SQPStatus status)
{
  return status == SQPStatus::kIterationLimit || status == SQPStatus::kTimeLimit ||
         status == SQPStatus::kQPSolveFailed || status == SQPStatus::kStoppedByCallback;
}
}  // namespace

const bool SUPER_DEBUG_MODE = false;

TrustRegionSQPSolver::TrustRegionSQPSolver(QPSolver::Ptr qp_solver) : qp_solver(std::move(qp_solver)) {}

bool TrustRegionSQPSolver::init(QPProblem::Ptr qp_prob)
{
  qp_problem = std::move(qp_prob);

  // Initialize optimization parameters
  results_ = SQPResults(qp_problem->getNumNLPVars(), qp_problem->getNumNLPConstraints(), qp_problem->getNumNLPCosts());
  results_.best_var_vals = qp_problem->getVariableValues();
  results_.merit_error_coeffs =
      Eigen::VectorXd::Constant(qp_problem->getNumNLPConstraints(), params.initial_merit_error_coeff);

  results_.best_costs = qp_problem->getExactCosts();

  results_.best_constraint_violations = qp_problem->getExactConstraintViolations();

  setBoxSize(params.initial_trust_box_size);
  constraintMeritCoeffChanged();
  return true;
}

void TrustRegionSQPSolver::setBoxSize(double box_size)
{
  qp_problem->setBoxSize(Eigen::VectorXd::Constant(qp_problem->getNumNLPVars(), box_size));
  results_.box_size = qp_problem->getBoxSize();
}

void TrustRegionSQPSolver::constraintMeritCoeffChanged()
{
  qp_problem->setConstraintMeritCoeff(results_.merit_error_coeffs);

  // Recalculate the best exact merit because merit coeffs may have changed
  results_.best_exact_merit =
      meritValue(results_.best_costs, results_.best_constraint_violations, results_.merit_error_coeffs);
}

void TrustRegionSQPSolver::registerCallback(const SQPCallback::Ptr& callback) { callbacks_.push_back(callback); }

const SQPStatus& TrustRegionSQPSolver::getStatus() { return status_; }

const SQPResults& TrustRegionSQPSolver::getResults() { return results_; }

void TrustRegionSQPSolver::solve(const QPProblem::Ptr& qp_problem)
{
  status_ = SQPStatus::kRunning;

  // Start time
  using Clock = std::chrono::steady_clock;
  auto start_time = Clock::now();

  // Initialize solver
  init(qp_problem);

  // A merit that is not finite at the start cannot rank any step
  if (!std::isfinite(results_.best_exact_merit))
  {
    TESSERACT_LOG_ERROR("The merit at the start point is not finite ({})", results_.best_exact_merit);
    status_ = SQPStatus::kNonFiniteMerit;
    results_.best_is_feasible = bestIsFeasible();
    qp_problem->setVariables(results_.best_var_vals.data());
    return;
  }

  // Penalty Iteration Loop
  for (int penalty_iteration = 0; penalty_iteration < params.max_merit_coeff_increases; penalty_iteration++)
  {
    results_.penalty_iteration = penalty_iteration;
    results_.convexify_iteration = 0;

    // Convexification loop: bounded by max_iterations and max_time, since every round solves at least one QP
    while (true)
    {
      const double elapsed_time = std::chrono::duration<double, std::milli>(Clock::now() - start_time).count() / 1000.0;
      if (elapsed_time > params.max_time)
      {
        TESSERACT_LOG_DEBUG("Elapsed time {} has exceeded max time {}", elapsed_time, params.max_time);
        status_ = SQPStatus::kTimeLimit;
        break;
      }

      if (results_.overall_iteration >= params.max_iterations)
      {
        TESSERACT_LOG_DEBUG("Iteration limit");
        status_ = SQPStatus::kIterationLimit;
        break;
      }

      if (stepSQPSolver())
        break;
    }

    if (isTerminal(status_))
      break;

    // Check if constraints are satisfied
    if (verifySQPSolverConvergence())
    {
      status_ = SQPStatus::kConverged;
      break;
    }

    // Set status to running
    status_ = SQPStatus::kRunning;

    // ---------------------------
    // Constraints are not satisfied!
    // Penalty Adjustment
    // ---------------------------
    adjustPenalty();
  }  // Penalty adjustment loop

  // If status is still set to running the penalty iteration limit was reached
  if (status_ == SQPStatus::kRunning)
  {
    status_ = SQPStatus::kPenaltyIterationLimit;
    TESSERACT_LOG_DEBUG("Penalty iteration limit, optimization couldn't satisfy all constraints");
  }

  results_.best_is_feasible = bestIsFeasible();

  // Final Cleanup
  if (SUPER_DEBUG_MODE)
    results_.print();

  qp_problem->setVariables(results_.best_var_vals.data());
}

bool TrustRegionSQPSolver::bestIsFeasible() const
{
  const Eigen::VectorXd& raw = results_.best_constraint_violations.raw;
  return raw.size() == 0 || (raw.allFinite() && raw.maxCoeff() < params.cnt_tolerance);
}

double TrustRegionSQPSolver::certifiedGap() const
{
  if (last_qp_status_ != QPSolveStatus::kSolved)
    return std::numeric_limits<double>::infinity();
  const double gap = qp_solver->getDualityGap();
  return std::isnan(gap) ? std::numeric_limits<double>::infinity() : gap;
}

void TrustRegionSQPSolver::pushTrustRegion()
{
  qp_solver->updateBounds(qp_problem->getBoundsLower(), qp_problem->getBoundsUpper());
  results_.box_size = qp_problem->getBoxSize();
}

void TrustRegionSQPSolver::scaleTrustRegion(double ratio)
{
  qp_problem->scaleBoxSize(ratio);
  pushTrustRegion();
}

void TrustRegionSQPSolver::shrinkTrustRegion(bool uncertified)
{
  scaleTrustRegion(params.trust_shrink_ratio);
  if (uncertified)
    uncertified_rejection_ = true;
}

void TrustRegionSQPSolver::recordInnerExit(SQPExitReason reason)
{
  results_.exit_reason = reason;
  results_.tiny_trust_region_after_uncertified = (reason == SQPExitReason::kTinyTrustRegion && uncertified_rejection_);
}

bool TrustRegionSQPSolver::verifySQPSolverConvergence()
{
  if (!bestIsFeasible())
    return false;

  if (results_.best_constraint_violations.raw.size() == 0)
    TESSERACT_LOG_DEBUG("Optimization has converged and there are no constraints");
  else
    TESSERACT_LOG_DEBUG("woo-hoo! constraints are satisfied (to tolerance {:.2e})", params.cnt_tolerance);
  return true;
}

void TrustRegionSQPSolver::adjustPenalty()
{
  if (params.inflate_constraints_individually)
  {
    assert(results_.best_constraint_violations.raw.size() == results_.merit_error_coeffs.size());
    for (Eigen::Index idx = 0; idx < results_.best_constraint_violations.raw.size(); idx++)
    {
      if (results_.best_constraint_violations.raw[idx] > params.cnt_tolerance)
      {
        TESSERACT_LOG_DEBUG("Not all constraints are satisfied. Increasing constraint penalties for {}", idx);
        results_.merit_error_coeffs[idx] *= params.merit_coeff_increase_ratio;
      }
    }
  }
  else
  {
    TESSERACT_LOG_DEBUG("Not all constraints are satisfied. Increasing constraint penalties uniformly");
    results_.merit_error_coeffs *= params.merit_coeff_increase_ratio;
  }
  setBoxSize(fmax(results_.box_size[0], params.min_trust_box_size / params.trust_shrink_ratio * 1.5));
  constraintMeritCoeffChanged();
}

bool TrustRegionSQPSolver::stepSQPSolver()
{
  results_.convexify_iteration++;

  const auto prev_nv = qp_problem->getNumQPVars();
  const auto prev_nc = qp_problem->getNumQPConstraints();

  qp_problem->convexify();

  const auto nv = qp_problem->getNumQPVars();
  const auto nc = qp_problem->getNumQPConstraints();

  const bool first_time = qp_solver->getSolverStatus() == QPSolverStatus::kUninitialized;
  const bool dims_changed = (nv != prev_nv || nc != prev_nc);

  if (first_time || dims_changed)
  {
    qp_solver->clear();
    qp_solver->init(nv, nc);
    qp_solver->updateHessianMatrix(qp_problem->getHessian());
    qp_solver->updateGradient(qp_problem->getGradient());
    qp_solver->updateLinearConstraintsMatrix(qp_problem->getConstraintMatrix());
    qp_solver->updateBounds(qp_problem->getBoundsLower(), qp_problem->getBoundsUpper());
    qp_solver->setWarmStart(*qp_problem);
  }
  else
  {
    // try update-in-place
    if (!qp_solver->updateHessianMatrix(qp_problem->getHessian()) ||
        !qp_solver->updateGradient(qp_problem->getGradient()) ||
        !qp_solver->updateLinearConstraintsMatrix(qp_problem->getConstraintMatrix()) ||
        !qp_solver->updateBounds(qp_problem->getBoundsLower(), qp_problem->getBoundsUpper()))
    {
      // pattern likely changed; fall back to full rebuild
      qp_solver->clear();
      qp_solver->init(nv, nc);
      qp_solver->updateHessianMatrix(qp_problem->getHessian());
      qp_solver->updateGradient(qp_problem->getGradient());
      qp_solver->updateLinearConstraintsMatrix(qp_problem->getConstraintMatrix());
      qp_solver->updateBounds(qp_problem->getBoundsLower(), qp_problem->getBoundsUpper());
      qp_solver->setWarmStart(*qp_problem);
    }
  }

  // Trust region loop
  runTrustRegionLoop();

  // A converged inner loop, a spent QP failure budget and a callback stop each end this convexification
  if (status_ == SQPStatus::kConverged || isTerminal(status_))
    return true;

  if (results_.box_size.maxCoeff() < params.min_trust_box_size)
  {
    TESSERACT_LOG_DEBUG("Converged because trust region is tiny");
    recordInnerExit(SQPExitReason::kTinyTrustRegion);
    status_ = SQPStatus::kConverged;
    return true;
  }
  return false;
}

void TrustRegionSQPSolver::runTrustRegionLoop()
{
  results_.trust_region_iteration = 0;
  uncertified_rejection_ = false;
  int qp_solver_failures = 0;
  while (results_.box_size.maxCoeff() >= params.min_trust_box_size)
  {
    if (SUPER_DEBUG_MODE)
      qp_problem->print();

    results_.overall_iteration++;
    results_.trust_region_iteration++;

    // Solve the current QP problem
    status_ = solveQPProblem();

    if (status_ == SQPStatus::kStoppedByCallback)
      return;  // Respect callbacks and exit gracefully

    if (status_ != SQPStatus::kRunning)
    {
      qp_solver_failures++;
      TESSERACT_LOG_WARN("Convex solver failed ({}/{})!", qp_solver_failures, params.max_qp_solver_failures);

      if (qp_solver_failures < params.max_qp_solver_failures)
      {
        shrinkTrustRegion(true);
        TESSERACT_LOG_DEBUG("Shrunk trust region. New box size: {:.4f}", results_.box_size[0]);
        status_ = SQPStatus::kRunning;
        continue;
      }

      if (qp_solver_failures == params.max_qp_solver_failures)
      {
        // Convex solver failed and this is the last attempt so setting the trust region to the minimum
        qp_problem->setBoxSize(Eigen::VectorXd::Constant(qp_problem->getNumNLPVars(), params.min_trust_box_size));
        pushTrustRegion();

        TESSERACT_LOG_DEBUG("Shrunk trust region to minimum. New box size: {:.4f}", results_.box_size[0]);
        status_ = SQPStatus::kRunning;
        uncertified_rejection_ = true;
        continue;
      }

      TESSERACT_LOG_ERROR("The convex solver failed you one too many times.");
      return;
    }

    // A non-finite merit cannot be compared; reject the step
    if (!std::isfinite(results_.new_exact_merit) || !std::isfinite(results_.new_approx_merit))
    {
      TESSERACT_LOG_WARN("Merit at the trial point is not finite (exact {}, approximate {}); rejecting the step",
                         results_.new_exact_merit,
                         results_.new_approx_merit);
      shrinkTrustRegion(true);
      continue;
    }

    // The best improvement the model offers lies in [approx, approx + gap]; exit only when its upper end is small
    const double gap = certifiedGap();
    // With min_approx_improve <= 0 no finite gap can block the small-improvement exit; only a non-finite one counts
    const bool uncertified = !std::isfinite(gap) || (params.min_approx_improve > 0 && gap >= params.min_approx_improve);
    const double approx = results_.approx_merit_improve;
    const double denom = std::max(std::abs(results_.best_exact_merit), 1e-12);
    const double roundoff = 1e-12 * std::max(1.0, std::abs(results_.best_exact_merit));
    const double certified_ratio = (approx + gap) / denom;

    if (approx < -(gap + roundoff))
    {
      TESSERACT_LOG_WARN("QP predicted a merit increase beyond its duality gap ({:.3e} < -{:.3e})", approx, gap);
    }
    else if (approx + gap < params.min_approx_improve)
    {
      TESSERACT_LOG_DEBUG("Converged because improvement was small ({:.3e} + gap {:.3e} < {:.3e})",
                          approx,
                          gap,
                          params.min_approx_improve);
      recordInnerExit(SQPExitReason::kSmallImprovement);
      status_ = SQPStatus::kConverged;
      return;
    }
    else if (certified_ratio < params.min_approx_improve_frac)
    {
      TESSERACT_LOG_DEBUG("Converged because improvement ratio was small ({:.3e} < {:.3e})",
                          certified_ratio,
                          params.min_approx_improve_frac);
      recordInnerExit(SQPExitReason::kSmallImprovementRatio);
      status_ = SQPStatus::kConverged;
      return;
    }

    if (approx < params.min_approx_improve || approx / denom < params.min_approx_improve_frac)
      ++results_.n_suppressed_exits;

    // Check if the bounding trust region needs to be shrunk
    // This happens if the exact solution got worse or if the QP approximation deviates from the exact by too much
    if (results_.exact_merit_improve < 0 || results_.merit_improve_ratio < params.improve_ratio_threshold)
    {
      shrinkTrustRegion(uncertified);
      TESSERACT_LOG_DEBUG("Shrunk trust region. new box size: {:.4f}", results_.box_size[0]);
    }
    else
    {
      results_.best_var_vals = results_.new_var_vals.head(qp_problem->getNumNLPVars());

      results_.best_exact_merit = results_.new_exact_merit;
      results_.best_constraint_violations = results_.new_constraint_violations;
      results_.best_costs = results_.new_costs;

      results_.best_approx_merit = results_.new_approx_merit;
      results_.best_approx_constraint_violations = results_.new_approx_constraint_violations;
      results_.best_approx_costs = results_.new_approx_costs;

      if (SUPER_DEBUG_MODE)
        results_.print();

      qp_problem->setVariables(results_.best_var_vals.data());

      scaleTrustRegion(params.trust_expand_ratio);
      TESSERACT_LOG_DEBUG("Expanded trust region. new box size: {:.4f}", results_.box_size[0]);
      return;
    }
  }  // Trust region loop
}

SQPStatus TrustRegionSQPSolver::solveQPProblem()
{
  // Solve the QP
  last_qp_status_ = qp_solver->solve();

  bool callbacks_ok = true;
  if (last_qp_status_ != QPSolveStatus::kFailed)
  {
    results_.new_var_vals = qp_solver->getSolution();

    // A non-finite solution is a failed solve; never evaluate it
    if (!results_.new_var_vals.allFinite())
    {
      TESSERACT_LOG_WARN("QP solver returned a non-finite solution; treating the solve as failed");
      qp_problem->setVariables(results_.best_var_vals.data());
      return SQPStatus::kQPSolveFailed;
    }

    if (last_qp_status_ == QPSolveStatus::kUnconverged)
    {
      ++results_.n_unconverged_qp_solves;

      // Joint limits and the trust box exist only as QP bounds; hold an unconverged solution to them
      const Eigen::Index n = qp_problem->getNumNLPVars();
      results_.new_var_vals.head(n) = results_.new_var_vals.head(n)
                                          .cwiseMax(qp_problem->getNLPVariableBoundsLower())
                                          .cwiseMin(qp_problem->getNLPVariableBoundsUpper());
    }

    // Calculate approximate QP merits (cheap)
    qp_problem->setVariables(results_.new_var_vals.data());

    // Evaluate convexified constraint violations (expensive)
    results_.new_approx_constraint_violations = qp_problem->evaluateConvexConstraintViolations(results_.new_var_vals);

    // Evaluate convexified costs (expensive)
    results_.new_approx_costs = qp_problem->evaluateConvexCosts(results_.new_var_vals);

    // Convexified merit
    results_.new_approx_merit =
        meritValue(results_.new_approx_costs, results_.new_approx_constraint_violations, results_.merit_error_coeffs);

    results_.approx_merit_improve = results_.best_exact_merit - results_.new_approx_merit;

    // Evaluate exact costs (expensive)
    results_.new_costs = qp_problem->getExactCosts();

    // Evaluate exact constraint violations (expensive)
    results_.new_constraint_violations = qp_problem->getExactConstraintViolations();

    // Calculate exact NLP merits (expensive) - TODO: Look into caching for qp_solver->Convexify()
    results_.new_exact_merit =
        meritValue(results_.new_costs, results_.new_constraint_violations, results_.merit_error_coeffs);
    results_.exact_merit_improve = results_.best_exact_merit - results_.new_exact_merit;
    // results_.merit_improve_ratio = results_.exact_merit_improve / results_.approx_merit_improve;
    if (std::abs(results_.approx_merit_improve) < 1e-12)
      results_.merit_improve_ratio = 0.0;  // or 1.0, or whatever convention you want
    else
      results_.merit_improve_ratio = results_.exact_merit_improve / results_.approx_merit_improve;

    // The variable are changed to the new values to calculated data but must be set
    // to best var vals because the new values may not improve the merit which is determined later.
    qp_problem->setVariables(results_.best_var_vals.data());

    // Print debugging info
    if (verbose)
      printStepInfo();

    // Call callbacks
    callbacks_ok = callCallbacks();
  }
  else
  {
    qp_problem->setVariables(results_.best_var_vals.data());

    TESSERACT_LOG_ERROR("Solver Failure");
    return SQPStatus::kQPSolveFailed;
  }

  // Check if any callbacks returned false
  if (!callbacks_ok)
  {
    return SQPStatus::kStoppedByCallback;
  }

  return SQPStatus::kRunning;
}

bool TrustRegionSQPSolver::callCallbacks()
{
  bool success = true;
  for (const auto& callback : callbacks_)
    success &= callback->execute(*qp_problem, results_);
  return success;
}

void TrustRegionSQPSolver::printStepInfo() const
{
  // Print Header
  std::printf("\n| %s |\n", std::string(88, '=').c_str());
  std::printf("| %s %s %s |\n", std::string(36, ' ').c_str(), "ROS Industrial", std::string(36, ' ').c_str());
  std::printf(
      "| %s %s %s |\n", std::string(28, ' ').c_str(), "TrajOpt Ifopt Motion Planning", std::string(29, ' ').c_str());
  std::printf("| %s |\n", std::string(88, '=').c_str());
  std::printf("| %s %s (Box Size: %-3.9f) %s |\n",
              std::string(26, ' ').c_str(),
              "Iteration",
              results_.box_size(0),
              std::string(27, ' ').c_str());
  std::printf("| %s |\n", std::string(88, '-').c_str());
  std::printf("| %14s: %-4d | %14s: %-4d | %15s: %-3d | %14s: %-3d |\n",
              "Overall",
              results_.overall_iteration,
              "Convexify",
              results_.convexify_iteration,
              "Trust Region",
              results_.trust_region_iteration,
              "Penalty",
              results_.penalty_iteration);
  std::printf("| %s |\n", std::string(88, '=').c_str());

  // Print Cost and Constraint Data
  std::printf("| %10s | %10s | %10s | %10s | %10s | %10s | %10s |\n",
              "merit",
              "oldexact",
              "new_exact",
              "new_approx",
              "dapprox",
              "dexact",
              "ratio");

  // Individual Costs
  std::printf("| %s | INDIVIDUAL COSTS\n", std::string(88, '-').c_str());
  // Loop over costs
  const std::vector<std::string>& cost_names = qp_problem->getNLPCostNames();
  for (Eigen::Index cost_number = 0; cost_number < static_cast<Eigen::Index>(cost_names.size()); ++cost_number)
  {
    const double approx_improve = results_.best_costs[cost_number] - results_.new_approx_costs[cost_number];
    const double exact_improve = results_.best_costs[cost_number] - results_.new_costs[cost_number];
    if (fabs(approx_improve) > 1e-8)
      std::printf("| %10s | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %-15s\n",
                  "----------",
                  results_.best_costs[cost_number],
                  results_.new_costs[cost_number],
                  results_.new_approx_costs[cost_number],
                  approx_improve,
                  exact_improve,
                  exact_improve / approx_improve,
                  cost_names[static_cast<std::size_t>(cost_number)].c_str());
    else
      std::printf("| %10s | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10s | %-15s\n",
                  "----------",
                  results_.best_costs[cost_number],
                  results_.new_costs[cost_number],
                  results_.new_approx_costs[cost_number],
                  approx_improve,
                  exact_improve,
                  "  ------  ",
                  cost_names[static_cast<std::size_t>(cost_number)].c_str());
  }

  // Sum Cost
  std::printf("| %s |\n", std::string(88, '=').c_str());
  std::printf("| %10s | %10.3e | %10.3e | %10.3e | %10s | %10s | %10s | SUM COSTS\n",
              "----------",
              results_.best_costs.sum(),
              results_.new_costs.sum(),
              results_.new_approx_costs.sum(),
              "----------",
              "----------",
              "----------");
  std::printf("| %s |\n", std::string(88, '=').c_str());

  // Individual Constraints
  // If we want to print the names we will have to add a getConstraints function to IFOPT
  if (results_.new_constraint_violations.raw.size() != 0)
  {
    std::printf("| %s | CONSTRAINTS\n", std::string(88, '-').c_str());
    const std::vector<std::string>& constraint_names = qp_problem->getNLPConstraintNames();
    // Loop over constraints
    for (Eigen::Index cnt_number = 0; cnt_number < static_cast<Eigen::Index>(constraint_names.size()); ++cnt_number)
    {
      // Each column is a merit contribution: the merit coefficient times a weighted violation.
      const double mu = results_.merit_error_coeffs[cnt_number];
      const double best = results_.best_constraint_violations.weighted[cnt_number];
      const double new_exact = results_.new_constraint_violations.weighted[cnt_number];
      const double new_approx = results_.new_approx_constraint_violations.weighted[cnt_number];
      const double approx_improve = best - new_approx;
      const double exact_improve = best - new_exact;
      if (fabs(approx_improve) > 1e-8)
        std::printf("| %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %-15s\n",
                    mu,
                    mu * best,
                    mu * new_exact,
                    mu * new_approx,
                    mu * approx_improve,
                    mu * exact_improve,
                    exact_improve / approx_improve,
                    constraint_names[static_cast<std::size_t>(cnt_number)].c_str());
      else
        std::printf("| %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10.3e | %10s | %-15s \n",
                    mu,
                    mu * best,
                    mu * new_exact,
                    mu * new_approx,
                    mu * approx_improve,
                    mu * exact_improve,
                    "  ------  ",
                    constraint_names[static_cast<std::size_t>(cnt_number)].c_str());
    }
  }

  // Constraint
  const Eigen::VectorXd& new_violations = results_.new_constraint_violations.raw;
  const std::string constraints_satisfied =
      (new_violations.size() == 0 || new_violations.maxCoeff() < params.cnt_tolerance) ? "True" : "False";
  std::printf("| %s |\n", std::string(88, '=').c_str());
  std::printf("| %10s | %10.3e | %10.3e | %10.3e | %10s | %10s | %10s | SUM CONSTRAINTS (WITHOUT MERIT), Satisfied "
              "(%s)\n",
              "----------",
              results_.best_constraint_violations.raw.sum(),
              results_.new_constraint_violations.raw.sum(),
              results_.new_approx_constraint_violations.raw.sum(),
              "----------",
              "----------",
              "----------",
              constraints_satisfied.c_str());

  // Total
  std::printf("| %s |\n", std::string(88, '=').c_str());
  std::printf("| %10s | %10.3e | %10.3e | %10s | %10.3e | %10.3e | %10.3e | TOTAL = SUM COSTS + SUM CONSTRAINTS (WITH "
              "MERIT)\n",
              "----------",
              results_.best_exact_merit,
              results_.new_exact_merit,
              "----------",
              results_.approx_merit_improve,
              results_.exact_merit_improve,
              results_.merit_improve_ratio);
  std::printf("| %s |\n", std::string(88, '=').c_str());
}

}  // namespace trajopt_sqp
