#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <tesseract/common/eigen_types.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/types.h>
#include <tesseract/environment/environment.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/kinematics/utils.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_common/collision_types.h>
#include <trajopt_common/collision_utils.h>

using namespace tesseract::collision;
using namespace tesseract::common;
using namespace tesseract::environment;
using namespace tesseract::kinematics;

// The two endpoints of one trajectory segment, and the sub-segment interval a subdivided check
// reports a contact from. cc_time is global to the segment and falls inside that interval.
namespace
{
const std::string kActiveLink = "r_wrist_roll_link";
const std::string kSecondActiveLink = "r_elbow_flex_link";
constexpr double kSubStart = 0.30;
constexpr double kSubEnd = 0.50;
constexpr double kCcTime = 0.40;
constexpr double kMargin = 0.025;
constexpr double kMarginBuffer = 20.0;

Eigen::VectorXd segmentStart() { return (Eigen::VectorXd(7) << -1.1, 1.2, -1.5, -1.4, -1.1, -1.3, 0.2).finished(); }
Eigen::VectorXd segmentEnd() { return (Eigen::VectorXd(7) << -0.3, 0.7, -0.9, -0.8, -0.4, -0.7, 0.9).finished(); }
Eigen::VectorXd lerp(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double t) { return a + (b - a) * t; }
}  // namespace

class CollisionGradientFrameTest : public testing::Test
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>();
  JointGroup::ConstPtr manip_;
  const LinkId link_{ kActiveLink };
  const LinkId second_link_{ kSecondActiveLink };
  const Eigen::VectorXd q0_{ segmentStart() };
  const Eigen::VectorXd q1_{ segmentEnd() };

  void SetUp() override
  {
    const std::filesystem::path urdf_file(std::string(TRAJOPT_DATA_DIR) + "/arm_around_table.urdf");
    const std::filesystem::path srdf_file(std::string(TRAJOPT_DATA_DIR) + "/pr2.srdf");

    const ResourceLocator::Ptr locator = std::make_shared<GeneralResourceLocator>();
    ASSERT_TRUE(env_->init(urdf_file, srdf_file, locator));

    manip_ = env_->getJointGroup("right_arm");
    ASSERT_TRUE(manip_ != nullptr);
    ASSERT_TRUE(manip_->isActiveLinkId(link_));
    ASSERT_TRUE(manip_->isActiveLinkId(second_link_));
  }

  /** @brief A contact on an active link whose stored poses are those of the given configurations,
   * independent of the configurations the gradient will be linearised about. */
  ContactResult makeContact(const Eigen::VectorXd& pose_source,
                            const Eigen::VectorXd& cc_pose_source,
                            double cc_time,
                            ContinuousCollisionType cc_type) const
  {
    ContactResult cr;
    cr.link_ids[0] = link_;
    cr.link_ids[1] = env_->getRootLinkId();
    cr.nearest_points_local[0] = Eigen::Vector3d(0.06, -0.04, 0.03);
    cr.nearest_points_local[1] = Eigen::Vector3d::Zero();
    cr.transform[0] = manip_->calcFwdKin(pose_source).at(link_);
    cr.cc_transform[0] = manip_->calcFwdKin(cc_pose_source).at(link_);
    cr.cc_time[0] = cc_time;
    cr.cc_type[0] = cc_type;
    cr.cc_time[1] = cc_time;
    cr.cc_type[1] = cc_type;
    cr.normal = Eigen::Vector3d(0.0, 0.0, 1.0);
    cr.distance = -0.01;
    return cr;
  }

  /** @brief The gradient a correct implementation must return for link A: the numerical derivative
   * of the witness point's world position at q_jac, contracted with the contact normal. */
  Eigen::VectorXd referenceGradient(const ContactResult& cr, const Eigen::VectorXd& q_jac, std::size_t i = 0) const
  {
    Eigen::MatrixXd num_jac(6, manip_->numJoints());
    numericalJacobian(
        num_jac, Eigen::Isometry3d::Identity(), *manip_, q_jac, cr.link_ids[i], cr.nearest_points_local[i]);
    return ((i == 0) ? -1.0 : 1.0) * cr.normal.transpose() * num_jac.topRows(3);
  }

  /** @brief Compare a gradient against the reference, refusing to compare two vanishing vectors.
   * isApprox is a relative test and is satisfied by any pair of near-zero vectors, so a
   * configuration that produced no usable gradient would satisfy every comparison in this file
   * whatever the implementation does. The floor is what makes a pass mean something. */
  void expectMatchesReference(const Eigen::VectorXd& actual,
                              const ContactResult& cr,
                              const Eigen::VectorXd& q_jac,
                              std::size_t i = 0) const
  {
    const Eigen::VectorXd expected = referenceGradient(cr, q_jac, i);
    EXPECT_GT(expected.norm(), 1e-2) << "the configuration produces no usable gradient, so the "
                                        "comparison below would hold for any implementation";
    EXPECT_TRUE(actual.isApprox(expected, 1e-4))
        << "got      " << actual.transpose() << "\nexpected " << expected.transpose();
  }
};

// A contact reported between the two segment endpoints is linearised at the configuration
// interpolated by cc_time, so the reference point must be rotated by that configuration's pose and
// not by the sub-segment pose the collision check happened to store. Both halves are linearised
// about that same configuration and so take the same rotation: the stored transform belongs to the
// timestep0 half no more than the stored cc_transform belongs to the timestep1 half.
TEST_F(CollisionGradientFrameTest, BetweenContactGradientsAtInterpolatedState)  // NOLINT
{
  const ContactResult cr =
      makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime, ContinuousCollisionType::CCType_Between);

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q0_, q1_, cr, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  ASSERT_TRUE(results.cc_gradients[0].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
  expectMatchesReference(results.cc_gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
}

// An untyped contact on an active link is interpolated by cc_time exactly as a typed one is, so it
// takes the interpolated pose too. The absence of a continuous collision type weights the gradient
// differently; it does not move the configuration the gradient is linearised about.
TEST_F(CollisionGradientFrameTest, UntypedContactOnActiveLinkUsesInterpolatedState)  // NOLINT
{
  const ContactResult cr =
      makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime, ContinuousCollisionType::CCType_None);

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q0_, q1_, cr, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  ASSERT_TRUE(results.cc_gradients[0].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
  expectMatchesReference(results.cc_gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
}

// A contact pinned to a segment endpoint is linearised at that endpoint, so both halves take that
// endpoint's pose - including the timestep1 half, whose stored cc_transform is the far endpoint.
TEST_F(CollisionGradientFrameTest, EndpointContactGradientsUseTheEndpointState)  // NOLINT
{
  const ContactResult at_t0 = makeContact(q0_, q1_, kCcTime, ContinuousCollisionType::CCType_Time0);
  const ContactResult at_t1 = makeContact(q0_, q1_, kCcTime, ContinuousCollisionType::CCType_Time1);

  trajopt_common::GradientResults t0;
  trajopt_common::GradientResults t1;
  trajopt_common::getGradient(t0, q0_, q1_, at_t0, kMargin, kMarginBuffer, *manip_);
  trajopt_common::getGradient(t1, q0_, q1_, at_t1, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(t0.gradients[0].has_gradient);
  ASSERT_TRUE(t1.gradients[0].has_gradient);
  expectMatchesReference(t0.gradients[0].gradient, at_t0, q0_);
  expectMatchesReference(t0.cc_gradients[0].gradient, at_t0, q0_);
  expectMatchesReference(t1.gradients[0].gradient, at_t1, q1_);
  expectMatchesReference(t1.cc_gradients[0].gradient, at_t1, q1_);
}

// A contact between two active links carries one linearisation configuration for both, so the pose
// the second link is rotated by must be that configuration's, exactly as for the first.
TEST_F(CollisionGradientFrameTest, ContactBetweenTwoActiveLinksRotatesBothByTheSameState)  // NOLINT
{
  ContactResult cr =
      makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime, ContinuousCollisionType::CCType_Between);
  cr.link_ids[1] = second_link_;
  cr.nearest_points_local[1] = Eigen::Vector3d(-0.02, 0.05, 0.01);
  cr.transform[1] = manip_->calcFwdKin(lerp(q0_, q1_, kSubStart)).at(second_link_);
  cr.cc_transform[1] = manip_->calcFwdKin(lerp(q0_, q1_, kSubEnd)).at(second_link_);

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q0_, q1_, cr, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(results.gradients[1].has_gradient);
  ASSERT_TRUE(results.cc_gradients[1].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
  expectMatchesReference(results.gradients[1].gradient, cr, lerp(q0_, q1_, kCcTime), 1);
  expectMatchesReference(results.cc_gradients[1].gradient, cr, lerp(q0_, q1_, kCcTime), 1);
}

// Two active links need not share a linearisation configuration: a swept check times each link's
// contact independently, so each is rotated by the pose at its own cc_time.
TEST_F(CollisionGradientFrameTest, ContactBetweenTwoActiveLinksWithDistinctTimes)  // NOLINT
{
  constexpr double kOtherCcTime = 0.75;
  ContactResult cr =
      makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime, ContinuousCollisionType::CCType_Between);
  cr.link_ids[1] = second_link_;
  cr.nearest_points_local[1] = Eigen::Vector3d(-0.02, 0.05, 0.01);
  cr.transform[1] = manip_->calcFwdKin(lerp(q0_, q1_, kSubStart)).at(second_link_);
  cr.cc_transform[1] = manip_->calcFwdKin(lerp(q0_, q1_, kSubEnd)).at(second_link_);
  cr.cc_time[1] = kOtherCcTime;

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q0_, q1_, cr, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  ASSERT_TRUE(results.gradients[1].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
  expectMatchesReference(results.gradients[1].gradient, cr, lerp(q0_, q1_, kOtherCcTime), 1);
  expectMatchesReference(results.cc_gradients[1].gradient, cr, lerp(q0_, q1_, kOtherCcTime), 1);
}

// A discrete contact stores the pose of the state it was found at, so its gradient is already
// rotated by the configuration it is linearised about and must stay exactly where it is.
TEST_F(CollisionGradientFrameTest, DiscreteContactGradientIsUnchanged)  // NOLINT
{
  const Eigen::VectorXd q = lerp(q0_, q1_, kCcTime);
  const ContactResult cr = makeContact(q, q, 0.0, ContinuousCollisionType::CCType_None);

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q, cr, kMargin, kMarginBuffer, *manip_);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  EXPECT_FALSE(results.cc_gradients[0].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, q);
}

// The time weighting is a separate quantity from the reference frame and must not move.
TEST_F(CollisionGradientFrameTest, TimeWeightingIsUnchanged)  // NOLINT
{
  const ContactResult cr =
      makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime, ContinuousCollisionType::CCType_Between);

  trajopt_common::GradientResults results;
  trajopt_common::getGradient(results, q0_, q1_, cr, kMargin, kMarginBuffer, *manip_);

  EXPECT_NEAR(results.gradients[0].scale, 1.0 - kCcTime, 1e-12);
  EXPECT_NEAR(results.cc_gradients[0].scale, kCcTime, 1e-12);
  EXPECT_EQ(results.gradients[0].cc_type, ContinuousCollisionType::CCType_Between);
  EXPECT_EQ(results.cc_gradients[0].cc_type, ContinuousCollisionType::CCType_Between);
}
