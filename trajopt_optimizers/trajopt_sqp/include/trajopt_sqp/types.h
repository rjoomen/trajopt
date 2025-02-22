/**
 * @file types.h
 * @brief Contains types for the trust region sqp solver
 *
 * @author Matthew Powelson
 * @date May 18, 2020
 * @version TODO
 * @bug No known bugs
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
#ifndef TRAJOPT_SQP_TYPES_H_
#define TRAJOPT_SQP_TYPES_H_

#include <trajopt_sqp/eigen_types.h>

namespace trajopt_sqp
{
enum class ConstraintType : std::uint8_t
{
  EQ,
  INEQ
};

enum class CostPenaltyType : std::uint8_t
{
  SQUARED,
  ABSOLUTE,
  HINGE
};

/**
 * @brief This struct defines parameters for the SQP optimization. The optimization should not change this struct
 */
struct SQPParameters
{
  /** @brief Minimum ratio exact_improve/approx_improve to accept step */
  double improve_ratio_threshold = 0.25;
  /** @brief NLP converges if trust region is smaller than this */
  double min_trust_box_size = 1e-4;
  /** @brief NLP converges if approx_merit_improves is smaller than this */
  double min_approx_improve = 1e-4;
  /** @brief NLP converges if approx_merit_improve / best_exact_merit < min_approx_improve_frac */
  double min_approx_improve_frac = std::numeric_limits<double>::lowest();
  /** @brief Max number of QP calls allowed */
  int max_iterations = 50;

  /** @brief Trust region is scaled by this when it is shrunk */
  double trust_shrink_ratio = 0.1;
  /** @brief Trust region is expanded by this when it is expanded */
  double trust_expand_ratio = 1.5;

  /** @brief Any constraint under this value is not considered a violation */
  double cnt_tolerance = 1e-4;
  /** @brief Max number of times the constraints will be inflated */
  double max_merit_coeff_increases = 5;
  /** @brief Max number of times the QP solver can fail before optimization is aborted */
  int max_qp_solver_failures = 3;
  /** @brief Constraints are scaled by this amount when inflated */
  double merit_coeff_increase_ratio = 10;
  /** @brief Max time in seconds that the optimizer will run */
  double max_time = std::numeric_limits<double>::max();
  /** @brief Initial coefficient that is used to scale the constraints. The total constaint cost is constaint_value
   * coeff * merit_coeff */
  double initial_merit_error_coeff = 10;
  /** @brief If true, only the constraints that are violated will be inflated */
  bool inflate_constraints_individually = true;
  /** @brief Initial size of the trust region */
  double initial_trust_box_size = 1e-1;
  /** @brief Unused */
  bool log_results = false;
  /** @brief Unused */
  std::string log_dir = "/tmp";
};

/** @brief This struct contains information and results for the SQP problem */
struct SQPResults
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SQPResults() = default;
  SQPResults(Eigen::Index num_vars, Eigen::Index num_cnts, Eigen::Index num_costs);

  /** @brief The lowest cost ever achieved */
  double best_exact_merit{ std::numeric_limits<double>::max() };
  /** @brief The cost achieved this iteration */
  double new_exact_merit{ std::numeric_limits<double>::max() };
  /** @brief The lowest convexified cost ever achieved */
  double best_approx_merit{ std::numeric_limits<double>::max() };
  /** @brief The convexified cost achieved this iteration */
  double new_approx_merit{ std::numeric_limits<double>::max() };

  /** @brief Variable values associated with best_exact_merit */
  Eigen::VectorXd best_var_vals;
  /** @brief Variable values associated with this iteration */
  Eigen::VectorXd new_var_vals;

  /** @brief Amount the convexified cost improved over the best this iteration */
  double approx_merit_improve{ 0 };
  /** @brief Amount the exact cost improved over the best this iteration */
  double exact_merit_improve{ 0 };
  /** @brief The amount the cost improved as a ratio of the total cost */
  double merit_improve_ratio{ 0 };

  /** @brief Vector defing the box size. The box is var_vals +/- box_size */
  Eigen::VectorXd box_size;
  /** @brief Coefficients used to weight the constraint violations */
  Eigen::VectorXd merit_error_coeffs;

  /** @brief Vector of the constraint violations. Positive is a violation */
  Eigen::VectorXd best_constraint_violations;
  /** @brief Vector of the constraint violations. Positive is a violation */
  Eigen::VectorXd new_constraint_violations;

  /** @brief Vector of the convexified constraint violations. Positive is a violation */
  Eigen::VectorXd best_approx_constraint_violations;
  /** @brief Vector of the convexified constraint violations. Positive is a violation */
  Eigen::VectorXd new_approx_constraint_violations;

  /** @brief Vector of the constraint violations. Positive is a violation */
  Eigen::VectorXd best_costs;
  /** @brief Vector of the constraint violations. Positive is a violation */
  Eigen::VectorXd new_costs;

  /** @brief Vector of the convexified costs.*/
  Eigen::VectorXd best_approx_costs;
  /** @brief Vector of the convexified costs.*/
  Eigen::VectorXd new_approx_costs;

  /** @brief The names associated to constraint violations */
  std::vector<std::string> constraint_names;
  /** @brief The names associated to costs */
  std::vector<std::string> cost_names;

  int penalty_iteration{ 0 };
  int convexify_iteration{ 0 };
  int trust_region_iteration{ 0 };
  int overall_iteration{ 0 };

  void print() const;
};

/**
 * @brief Status codes for the SQP Optimization
 */
enum class SQPStatus : std::uint8_t
{
  RUNNING,                 /**< Optimization is currently running */
  NLP_CONVERGED,           /**< NLP Successfully converged */
  ITERATION_LIMIT,         /**< SQP Optimization reached iteration limit */
  PENALTY_ITERATION_LIMIT, /**< SQP Optimization reached penalty iteration limit */
  OPT_TIME_LIMIT,          /**< SQP Optimization reached reached limit */
  QP_SOLVER_ERROR,         /**< QP Solver failed */
  CALLBACK_STOPPED         /**< Optimization stopped because callback returned false */
};

}  // namespace trajopt_sqp

#endif
