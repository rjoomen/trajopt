/**
 * @file cartesian_position_constraint.h
 * @brief The cartesian position constraint
 *
 * @author Levi Armstrong
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
#ifndef TRAJOPT_IFOPT_CARTESIAN_POSITION_CONSTRAINT_H
#define TRAJOPT_IFOPT_CARTESIAN_POSITION_CONSTRAINT_H

#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Eigen>
#include <ifopt/constraint_set.h>
#include <tesseract_kinematics/core/fwd.h>
TRAJOPT_IGNORE_WARNINGS_POP

namespace trajopt_ifopt
{
class JointPosition;

/** @brief Contains Cartesian pose constraint information */
struct CartPosInfo
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<CartPosInfo>;
  using ConstPtr = std::shared_ptr<const CartPosInfo>;

  enum class Type : std::uint8_t
  {
    TARGET_ACTIVE,
    SOURCE_ACTIVE,
    BOTH_ACTIVE
  };

  CartPosInfo() = default;
  CartPosInfo(std::shared_ptr<const tesseract_kinematics::JointGroup> manip,
              std::string source_frame,
              std::string target_frame,
              const Eigen::Isometry3d& source_frame_offset = Eigen::Isometry3d::Identity(),
              const Eigen::Isometry3d& target_frame_offset = Eigen::Isometry3d::Identity(),
              const Eigen::VectorXi& indices = Eigen::Matrix<int, 1, 6>(std::vector<int>({ 0, 1, 2, 3, 4, 5 }).data()));

  /** @brief The joint group */
  std::shared_ptr<const tesseract_kinematics::JointGroup> manip;

  /** @brief Link which should reach desired pos */
  std::string source_frame;

  /** @brief The target frame that should be reached by the source */
  std::string target_frame;

  /** @brief Static transform applied to the source_frame location */
  Eigen::Isometry3d source_frame_offset;

  /** @brief Static transform applied to the target_frame location */
  Eigen::Isometry3d target_frame_offset;

  /** @brief indicates which link is active */
  Type type{ Type::TARGET_ACTIVE };

  /**
   * @brief This is a vector of indices to be returned Default: {0, 1, 2, 3, 4, 5}
   *
   * If you only care about x, y and z error, this is {0, 1, 2}
   * If you only care about rotation error around x, y and z, this is {3, 4, 5}
   */
  Eigen::VectorXi indices;
};

class CartPosConstraint : public ifopt::ConstraintSet
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<CartPosConstraint>;
  using ConstPtr = std::shared_ptr<const CartPosConstraint>;
  using ErrorFunctionType = std::function<Eigen::VectorXd(const Eigen::Isometry3d&, const Eigen::Isometry3d&)>;
  using ErrorDiffFunctionType =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::Isometry3d&, const Eigen::Isometry3d&)>;

  CartPosConstraint(const CartPosInfo& info,
                    std::shared_ptr<const JointPosition> position_var,
                    const std::string& name = "CartPos");

  CartPosConstraint(CartPosInfo info,
                    std::shared_ptr<const JointPosition> position_var,
                    const Eigen::VectorXd& coeffs,
                    const std::string& name = "CartPos");

  /**
   * @brief CalcValues Calculates the values associated with the constraint
   * @param joint_vals Input joint values for which FK is solved
   * @return Error of FK solution from target, size 6. The first 3 terms are associated with position and the last 3 are
   * associated with orientation.
   */
  Eigen::VectorXd CalcValues(const Eigen::Ref<const Eigen::VectorXd>& joint_vals) const;
  /**
   * @brief Returns the values associated with the constraint. In this case that is the concatenated joint values
   * associated with each of the joint positions should be n_dof_ * n_vars_ long
   * @return
   */
  Eigen::VectorXd GetValues() const override;

  /**
   * @brief  Returns the "bounds" of this constraint. How these are enforced is up to the solver
   * @return Returns the "bounds" of this constraint
   */
  std::vector<ifopt::Bounds> GetBounds() const override;

  void SetBounds(const std::vector<ifopt::Bounds>& bounds);

  /**
   * @brief Fills the jacobian block associated with the constraint
   * @param jac_block Block of the overall jacobian associated with these constraints
   */
  void CalcJacobianBlock(const Eigen::Ref<const Eigen::VectorXd>& joint_vals, Jacobian& jac_block) const;
  /**
   * @brief Fills the jacobian block associated with the given var_set.
   * @param var_set Name of the var_set to which the jac_block is associated
   * @param jac_block Block of the overall jacobian associated with these constraints and the var_set variable
   */
  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

  /**
   * @brief Gets the Cartesian Pose info used to create this constraint
   * @return The Cartesian Pose info used to create this constraint
   */
  const CartPosInfo& GetInfo() const;
  CartPosInfo& GetInfo();

  /**
   * @brief Set the target pose
   * @param pose
   */
  void SetTargetPose(const Eigen::Isometry3d& target_frame_offset);

  /**
   * @brief Returns the target pose for the constraint
   * @return The target pose for the constraint
   */
  Eigen::Isometry3d GetTargetPose() const;

  /**
   * @brief Returns the current TCP pose in world frame given the input kinematic info and the current variable values
   * @return The current TCP pose given the input kinematic info and the current variable values
   */
  Eigen::Isometry3d GetCurrentPose() const;

  /** @brief If true, numeric differentiation will be used. Default: true
   *
   * Note: While the logic for using the jacobian from KDL will be used if set to false, this has been buggy. Set this
   * to false at your own risk.
   */
  bool use_numeric_differentiation{ true };

private:
  /** @brief The number of joints in a single JointPosition */
  long n_dof_;

  /** @brief The constraint coefficients */
  Eigen::VectorXd coeffs_;

  /** @brief Bounds on the positions of each joint */
  std::vector<ifopt::Bounds> bounds_;

  /**
   * @brief Pointers to the vars used by this constraint.
   * Do not access them directly. Instead use this->GetVariables()->GetComponent(position_var->GetName())->GetValues()
   */
  std::shared_ptr<const JointPosition> position_var_;

  /** @brief The kinematic information used when calculating error */
  CartPosInfo info_;

  /** @brief Error function for calculating the error in the position given the source and target positions */
  ErrorFunctionType error_function_{ nullptr };

  /** @brief The error function to calculate the error difference used for jacobian calculations */
  ErrorDiffFunctionType error_diff_function_{ nullptr };
};
}  // namespace trajopt_ifopt
#endif
