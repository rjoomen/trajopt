#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <tesseract/common/eigen_types.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/types.h>
#include <tesseract/environment/environment.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/kinematics/utils.h>
#include <console_bridge/console.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/collision_terms.hpp>
#include <trajopt_common/logging.hpp>

using namespace trajopt;
using namespace tesseract::collision;
using namespace tesseract::common;
using namespace tesseract::environment;
using namespace tesseract::kinematics;

namespace
{
/** @brief Exposes the base evaluator's gradient entry point; the collision entry points are unused
 * here and are given the smallest bodies that satisfy the interface. */
struct GradientOnlyEvaluator : public CollisionEvaluator
{
  GradientOnlyEvaluator(const JointGroup::ConstPtr& manip, Environment::ConstPtr env)
    : CollisionEvaluator(manip, std::move(env), false)
  {
  }

  void CalcDistExpressions(const DblVec& /*x*/,
                           sco::AffExprVector& /*exprs*/,
                           std::vector<double>& /*exprs_margin*/,
                           std::vector<double>& /*exprs_coeff*/) override
  {
  }
  void CalcCollisions(const DblVec& /*x*/, ContactResultMap& /*dist_results*/) override {}
  void Plot(const std::shared_ptr<tesseract::visualization::Visualization>& /*plotter*/, const DblVec& /*x*/) override
  {
  }
  sco::VarVector GetVars() override { return {}; }
};
}  // namespace

// The two endpoints of one trajectory segment, and the sub-segment interval a subdivided check
// reports a contact from. cc_time is global to the segment and falls inside that interval.
namespace
{
const std::string kActiveLink = "r_wrist_roll_link";
constexpr double kSubStart = 0.30;
constexpr double kSubEnd = 0.50;
constexpr double kCcTime = 0.40;

Eigen::VectorXd segmentStart() { return (Eigen::VectorXd(7) << -1.1, 1.2, -1.5, -1.4, -1.1, -1.3, 0.2).finished(); }
Eigen::VectorXd segmentEnd() { return (Eigen::VectorXd(7) << -0.3, 0.7, -0.9, -0.8, -0.4, -0.7, 0.9).finished(); }
Eigen::VectorXd lerp(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double t) { return a + (b - a) * t; }
}  // namespace

class CastGradientFrameTest : public testing::Test
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>();
  JointGroup::ConstPtr manip_;
  std::shared_ptr<GradientOnlyEvaluator> evaluator_;
  const LinkId link_{ kActiveLink };
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
    evaluator_ = std::make_shared<GradientOnlyEvaluator>(manip_, env_);

    trajopt_common::gLogLevel = trajopt_common::LevelError;
  }

  /** @brief A contact on an active link, typed as occurring between two configurations, whose
   * stored poses are those of an arbitrary interval rather than of q_jac. */
  ContactResult makeContact(const Eigen::VectorXd& pose_source,
                            const Eigen::VectorXd& cc_pose_source,
                            double cc_time) const
  {
    ContactResult cr;
    cr.link_ids[0] = link_;
    cr.link_ids[1] = env_->getRootLinkId();
    cr.nearest_points_local[0] = Eigen::Vector3d(0.06, -0.04, 0.03);
    cr.nearest_points_local[1] = Eigen::Vector3d::Zero();
    cr.transform[0] = manip_->calcFwdKin(pose_source).at(link_);
    cr.cc_transform[0] = manip_->calcFwdKin(cc_pose_source).at(link_);
    cr.cc_time[0] = cc_time;
    cr.cc_type[0] = ContinuousCollisionType::CCType_Between;
    cr.normal = Eigen::Vector3d(0.0, 0.0, 1.0);
    cr.distance = -0.01;
    return cr;
  }

  /** @brief The gradient a correct implementation must return for link A: the numerical derivative
   * of the witness point's world position at q_jac, contracted with the contact normal. */
  Eigen::VectorXd referenceGradient(const ContactResult& cr, const Eigen::VectorXd& q_jac) const
  {
    Eigen::MatrixXd num_jac(6, manip_->numJoints());
    numericalJacobian(
        num_jac, Eigen::Isometry3d::Identity(), *manip_, q_jac, cr.link_ids[0], cr.nearest_points_local[0]);
    return -1.0 * cr.normal.transpose() * num_jac.topRows(3);
  }

  /** @brief Compare a gradient against the reference, refusing to compare two vanishing vectors.
   * isApprox is a relative test and is satisfied by any pair of near-zero vectors, so a
   * configuration that produced no usable gradient would satisfy every comparison in this file
   * whatever the implementation does. The floor is what makes a pass mean something. */
  void expectMatchesReference(const Eigen::VectorXd& actual,
                              const ContactResult& cr,
                              const Eigen::VectorXd& q_jac) const
  {
    const Eigen::VectorXd expected = referenceGradient(cr, q_jac);
    EXPECT_GT(expected.norm(), 1e-2) << "the configuration produces no usable gradient, so the "
                                        "comparison below would hold for any implementation";
    EXPECT_TRUE(actual.isApprox(expected, 1e-4))
        << "got      " << actual.transpose() << "\nexpected " << expected.transpose();
  }
};

TEST_F(CastGradientFrameTest, SubdividedContactGradientAtSegmentStart)  // NOLINT
{
  const ContactResult cr = makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime);

  const GradientResults results = evaluator_->GetGradient(q0_, cr, 0.025, 20.0, false);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, q0_);
}

TEST_F(CastGradientFrameTest, SubdividedContactGradientAtSegmentEnd)  // NOLINT
{
  const ContactResult cr = makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime);

  const GradientResults results = evaluator_->GetGradient(q1_, cr, 0.025, 20.0, true);

  ASSERT_TRUE(results.gradients[0].has_gradient);
  expectMatchesReference(results.gradients[0].gradient, cr, q1_);
}

// With the stored poses equal to the poses at the evaluation configurations — the state of an
// unsubdivided check — the gradient must match the reference exactly: an unsubdivided contact
// carries no separate linearisation pose, so there is nothing for the reference frame to correct.
TEST_F(CastGradientFrameTest, UnsubdividedContactGradientIsUnchanged)  // NOLINT
{
  const ContactResult cr = makeContact(q0_, q1_, kCcTime);

  const GradientResults at_start = evaluator_->GetGradient(q0_, cr, 0.025, 20.0, false);
  const GradientResults at_end = evaluator_->GetGradient(q1_, cr, 0.025, 20.0, true);

  ASSERT_TRUE(at_start.gradients[0].has_gradient);
  ASSERT_TRUE(at_end.gradients[0].has_gradient);
  expectMatchesReference(at_start.gradients[0].gradient, cr, q0_);
  expectMatchesReference(at_end.gradients[0].gradient, cr, q1_);
}

// The two-state entry point picks its own linearisation configuration from cc_time rather than
// taking one from the caller, so it must rotate the reference point by the pose at that
// configuration and not by either stored pose.
TEST_F(CastGradientFrameTest, TwoStateGradientAtInterpolatedState)  // NOLINT
{
  const ContactResult cr = makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime);

  const GradientResults at_start = evaluator_->GetGradient(q0_, q1_, cr, 0.025, 20.0, false);
  const GradientResults at_end = evaluator_->GetGradient(q0_, q1_, cr, 0.025, 20.0, true);

  ASSERT_TRUE(at_start.gradients[0].has_gradient);
  ASSERT_TRUE(at_end.gradients[0].has_gradient);
  expectMatchesReference(at_start.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
  expectMatchesReference(at_end.gradients[0].gradient, cr, lerp(q0_, q1_, kCcTime));
}

// The time weighting is a separate quantity from the reference frame and must not move.
TEST_F(CastGradientFrameTest, TimeWeightingIsUnchanged)  // NOLINT
{
  const ContactResult cr = makeContact(lerp(q0_, q1_, kSubStart), lerp(q0_, q1_, kSubEnd), kCcTime);

  EXPECT_NEAR(evaluator_->GetGradient(q0_, cr, 0.025, 20.0, false).gradients[0].scale, 1.0 - kCcTime, 1e-12);
  EXPECT_NEAR(evaluator_->GetGradient(q1_, cr, 0.025, 20.0, true).gradients[0].scale, kCcTime, 1e-12);
}
