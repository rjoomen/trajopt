#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <ctime>
#include <gtest/gtest.h>
#include <tesseract_common/types.h>
#include <tesseract_common/resource_locator.h>
#include <tesseract_kinematics/core/joint_group.h>
#include <tesseract_scene_graph/scene_state.h>
#include <tesseract_environment/environment.h>
#include <tesseract_environment/utils.h>
#include <console_bridge/console.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/plot_callback.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt_sco/optimizers.hpp>
#include <trajopt_common/config.hpp>
#include <trajopt_common/eigen_conversions.hpp>
#include <trajopt_common/logging.hpp>
#include <trajopt_common/stl_to_string.hpp>

#include <trajopt/kinematic_terms.hpp>
#include <trajopt_sco/num_diff.hpp>

using namespace trajopt;
using namespace std;
using namespace trajopt_common;
using namespace tesseract_environment;
using namespace tesseract_collision;
using namespace tesseract_kinematics;
using namespace tesseract_visualization;
using namespace tesseract_scene_graph;
using namespace tesseract_common;

class KinematicCostsTest : public testing::Test
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>(); /**< Trajopt Basic Environment */

  void SetUp() override
  {
    const std::filesystem::path urdf_file(std::string(TRAJOPT_DATA_DIR) + "/arm_around_table.urdf");
    const std::filesystem::path srdf_file(std::string(TRAJOPT_DATA_DIR) + "/pr2.srdf");

    const ResourceLocator::Ptr locator = std::make_shared<tesseract_common::GeneralResourceLocator>();
    EXPECT_TRUE(env_->init(urdf_file, srdf_file, locator));

    gLogLevel = trajopt_common::LevelError;
  }
};

namespace
{
std::string toString(const Eigen::MatrixXd& mat)
{
  std::stringstream ss;
  ss << mat;
  return ss.str();
}

void checkJacobian(const sco::VectorOfVector& f,
                   const sco::MatrixOfVector& dfdx,
                   const Eigen::VectorXd& values,
                   const double epsilon)
{
  const Eigen::MatrixXd numerical = sco::calcForwardNumJac(f, values, epsilon);
  const Eigen::MatrixXd analytical = dfdx(values);

  const bool pass = numerical.isApprox(analytical, 1e-5);
  EXPECT_TRUE(pass);
  if (!pass)
  {
    CONSOLE_BRIDGE_logError("Numerical:\n %s", toString(numerical).c_str());
    CONSOLE_BRIDGE_logError("Analytical:\n %s", toString(analytical).c_str());
  }
}
}  // namespace

TEST_F(KinematicCostsTest, CartPoseJacCalculator)  // NOLINT
{
  CONSOLE_BRIDGE_logDebug("KinematicCostsTest, CartPoseJacCalculator");

  const tesseract_kinematics::JointGroup::ConstPtr kin = env_->getJointGroup("right_arm");

  const std::string source_frame = env_->getRootLinkName();
  const std::string target_frame = "r_gripper_tool_frame";
  const Eigen::Isometry3d source_frame_offset = env_->getState().link_transforms.at(target_frame);
  const Eigen::Isometry3d target_frame_offset =
      Eigen::Isometry3d::Identity() * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ());

  Eigen::VectorXd values(7);
  values << -1.1, 1.2, -3.3, -1.4, 5.5, -1.6, 7.7;

  const CartPoseErrCalculator f(kin, source_frame, target_frame, source_frame_offset, target_frame_offset);
  const CartPoseJacCalculator dfdx(kin, source_frame, target_frame, source_frame_offset, target_frame_offset);
  checkJacobian(f, dfdx, values, 1.0e-5);
}

// This has known issues and is not being used. Disabled due to segfaults in CI
// TEST_F(KinematicCostsTest, DynamicCartPoseJacCalculator)  // NOLINT
//{
//  CONSOLE_BRIDGE_logDebug("KinematicCostsTest, DynamicCartPoseJacCalculator");

//  auto env = env_->getEnvironment();
//  auto kin = env_->getManipulatorManager()->getFwdKinematicSolver("full_body");
//  std::unordered_map<std::string, double> j;
//  j["l_elbow_flex_joint"] = -0.15;
//  env->setState(j);
//  auto world_to_base = env->getCurrentState()->link_transforms.at(kin->getBaseLinkName());
//  auto adjacency_map = std::make_shared<tesseract_environment::AdjacencyMap>(
//      env->getSceneGraph(), kin->getActiveLinkNames(), env->getCurrentState()->link_transforms);

//  std::string link = "r_gripper_tool_frame";
//  std::string target = "l_gripper_tool_frame";
//  Eigen::Isometry3d tcp = Eigen::Isometry3d::Identity();

//  Eigen::VectorXd values(15);
//  values.setZero();
//  std::vector<std::string> joint_names = kin->getJointNames();
//  for (std::size_t i = 0; i < 15; ++i)
//  {
//    if (joint_names[i] == "r_elbow_flex_joint" || joint_names[i] == "l_elbow_flex_joint")
//      values(static_cast<long>(i)) = -0.15;

//    if (joint_names[i] == "r_wrist_flex_joint" || joint_names[i] == "l_wrist_flex_joint")
//      values(static_cast<long>(i)) = -0.1;
//  }

//  DynamicCartPoseErrCalculator f(target, kin, adjacency_map, world_to_base, link, tcp, tcp);
//  DynamicCartPoseJacCalculator dfdx(target, kin, adjacency_map, world_to_base, link, tcp, tcp);
//  checkJacobian(f, dfdx, values, 1.0e-5);
//}

////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
