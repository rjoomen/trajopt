#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Core>
#include <gtest/gtest.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <cmath>
#include <functional>
#include <memory>
#include <optional>

#include <trajopt_sco/expr_op_overloads.hpp>
#include <trajopt_sco/expr_ops.hpp>
#include <trajopt_sco/modeling_utils.hpp>
#include <trajopt_sco/optimizers.hpp>
#include <trajopt_sco/sco_common.hpp>
#include <tesseract/common/logging.h>

#include "scripted-model.hpp"

using namespace sco;

namespace
{
int nan_evaluations = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

/** @brief Count a cost evaluation at a point that is not finite */
void countNonFinite(const Eigen::VectorXd& x)
{
  if (!x.allFinite())
    ++nan_evaluations;
}

double targetCost(const Eigen::VectorXd& x)
{
  countNonFinite(x);
  return sq(x(0) - 0.8) + sq(x(1) + 0.3);
}

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
    results.status = OPT_NON_FINITE_MERIT;
    EXPECT_FALSE(isUsable(results));
    results.status = INVALID;
    EXPECT_FALSE(isUsable(results));
  }
}

namespace
{
/** @brief Pulled toward 0.8, but not defined above 0.5 */
double partialCost(const Eigen::VectorXd& x)
{
  countNonFinite(x);
  return x(0) > 0.5 ? std::nan("") : sq(x(0) - 0.8) + sq(x(1));
}

OptProb::Ptr makeScriptedProblem(const std::shared_ptr<test::ScriptedModel>& model,
                                 double (*cost)(const Eigen::VectorXd&))
{
  auto prob = std::make_shared<test::ScriptedProb>(model);
  prob->createVariables({ "x0", "x1" }, { -1.0, -1.0 }, { 1.0, 1.0 });
  prob->addCost(std::make_shared<CostFromFunc>(ScalarOfVector::construct(cost), prob->getVars(), "cost"));
  return prob;
}
}  // namespace

TEST_F(ScoTermination, NonFiniteSolutionIsAFailedSolveAndNeverEvaluated)  // NOLINT
{
  nan_evaluations = 0;
  auto model = std::make_shared<test::ScriptedModel>(createModel(ModelType::OSQP));
  model->script.push_back({ std::nullopt, [](DblVec& x) { x[0] = std::nan(""); }, std::nullopt });
  BasicTrustRegionSQP solver(makeScriptedProblem(model, &targetCost));
  solver.initialize({ 0.0, 0.0 });
  solver.optimize();
  EXPECT_GT(model->solves, 1);
  EXPECT_EQ(nan_evaluations, 0);
  EXPECT_TRUE(std::isfinite(solver.x()[0]));
}

TEST_F(ScoTermination, NonFiniteTrialMeritIsRejected)  // NOLINT
{
  auto model = std::make_shared<test::ScriptedModel>(createModel(ModelType::OSQP));
  BasicTrustRegionSQP solver(makeScriptedProblem(model, &partialCost));
  solver.getParameters().trust_box_size = 1.0;  // the first step overshoots into the undefined region
  solver.initialize({ 0.0, 0.0 });
  solver.optimize();
  EXPECT_TRUE(std::isfinite(solver.results().total_cost));
  EXPECT_LE(solver.x()[0], 0.5);
}

TEST_F(ScoTermination, NonFiniteStartFailsFast)  // NOLINT
{
  auto model = std::make_shared<test::ScriptedModel>(createModel(ModelType::OSQP));
  BasicTrustRegionSQP solver(makeScriptedProblem(model, &partialCost));
  solver.initialize({ 0.9, 0.0 });
  EXPECT_EQ(solver.optimize(), OPT_NON_FINITE_MERIT);
  EXPECT_EQ(model->solves, 0);
  EXPECT_FALSE(isUsable(solver.results()));
}

namespace
{
/** @brief Minimum 3 at x0 = 3; the offset makes the merit negative everywhere near the start */
double offsetCost(const Eigen::VectorXd& x) { return sq(x(0) - 3.0) - 100.0; }
}  // namespace

TEST_F(ScoTermination, FractionalExitUsesTheMeritMagnitude)  // NOLINT
{
  auto prob = std::make_shared<OptProb>(ModelType::OSQP);
  prob->createVariables({ "x0" }, { -10.0 }, { 10.0 });
  prob->addCost(std::make_shared<CostFromFunc>(ScalarOfVector::construct(&offsetCost), prob->getVars(), "offset"));
  BasicTrustRegionSQP solver(prob);
  solver.getParameters().min_approx_improve_frac = 1e-3;
  solver.getParameters().trust_box_size = 1.0;
  solver.initialize({ 0.0 });
  solver.optimize();
  // A positive improvement over a negative merit must not read as a negative ratio and end the run at the start
  EXPECT_NEAR(solver.x()[0], 3.0, 1e-2);
}

TEST_F(ScoTermination, LoggingToAMissingDirDoesNotCrash)  // NOLINT
{
  // fopen fails on a missing log_dir; optimize() must still return rather than fclose a null stream
  BasicTrustRegionSQP solver(makeProblem());
  solver.getParameters().log_results = true;
  solver.getParameters().log_dir = "/tmp/claude-1002/sco-termination-unit-missing-log-dir";
  solver.initialize({ 0.0, 0.0 });
  EXPECT_NO_FATAL_FAILURE(solver.optimize());
}

namespace
{
std::shared_ptr<test::ScriptedModel> scriptedOsqp()
{
  return std::make_shared<test::ScriptedModel>(createModel(ModelType::OSQP));
}

/** @brief The target cost, plus 1 at every point other than the start, with the uncharged cost as its exact model */
class ChargedTargetCost : public Cost
{
public:
  ChargedTargetCost(VarVector vars, DblVec start) : Cost("charged"), vars_(std::move(vars)), start_(std::move(start)) {}
  double value(const DblVec& x) override
  {
    const DblVec v{ vars_[0].value(x), vars_[1].value(x) };
    const double base = sq(v[0] - 0.8) + sq(v[1] + 0.3);
    return v == start_ ? base : base + 1.0;
  }
  ConvexObjective::Ptr convex(const DblVec& /*x*/, Model* model) override
  {
    auto out = std::make_shared<ConvexObjective>(model);
    out->addQuadExpr(exprSquare(exprSub(AffExpr(vars_[0]), 0.8)));
    out->addQuadExpr(exprSquare(exprAdd(AffExpr(vars_[1]), 0.3)));
    return out;
  }
  VarVector getVars() override { return vars_; }

private:
  VarVector vars_;
  DblVec start_;
};

OptResults runScripted(const std::shared_ptr<test::ScriptedModel>& model,
                       const std::function<void(BasicTrustRegionSQPParameters&)>& tune = nullptr)
{
  BasicTrustRegionSQP solver(makeScriptedProblem(model, &targetCost));
  if (tune)
    tune(solver.getParameters());
  solver.initialize({ 0.0, 0.0 });
  solver.optimize();
  return solver.results();
}
}  // namespace

TEST_F(ScoTermination, CertifiedSolvesExitAsBefore)  // NOLINT
{
  const OptResults r = runScripted(scriptedOsqp());
  EXPECT_EQ(r.status, OPT_CONVERGED);
  EXPECT_EQ(r.exit_reason, EXIT_SMALL_IMPROVEMENT);
  EXPECT_EQ(r.n_suppressed_exits, 0);
}

TEST_F(ScoTermination, LargeGapSuppressesTheExit)  // NOLINT
{
  auto model = scriptedOsqp();
  model->every_solve = test::ScriptedModelSolve{ std::nullopt, nullptr, 1.0 };
  const OptResults r = runScripted(model);
  EXPECT_NE(r.exit_reason, EXIT_SMALL_IMPROVEMENT);
  EXPECT_GT(r.n_suppressed_exits, 0);
}

TEST_F(ScoTermination, NaNGapIsUncertified)  // NOLINT
{
  auto model = scriptedOsqp();
  model->every_solve = test::ScriptedModelSolve{ std::nullopt, nullptr, std::nan("") };
  const OptResults r = runScripted(model);
  EXPECT_NE(r.exit_reason, EXIT_SMALL_IMPROVEMENT);
  EXPECT_GT(r.n_suppressed_exits, 0);
}

TEST_F(ScoTermination, PredictionBelowMinusGapGoesToTheRatioTest)  // NOLINT
{
  auto model = scriptedOsqp();
  model->script.push_back({ std::nullopt, [](DblVec& x) { x[0] = -0.05; }, 0.0 });
  const OptResults r = runScripted(model);
  EXPECT_EQ(r.status, OPT_CONVERGED);
  EXPECT_NEAR(r.x[0], 0.8, 1e-3);
  EXPECT_GE(r.n_suppressed_exits, 1);
}

TEST_F(ScoTermination, RatioExitRecordsItsReason)  // NOLINT
{
  const OptResults r = runScripted(scriptedOsqp(), [](BasicTrustRegionSQPParameters& p) {
    p.min_approx_improve = 1e-12;
    p.min_approx_improve_frac = 0.5;
  });
  EXPECT_EQ(r.exit_reason, EXIT_SMALL_IMPROVEMENT_RATIO);
}

TEST_F(ScoTermination, TinyBoxAfterUncertifiedRejectionsIsFlagged)  // NOLINT
{
  auto run = [](double gap) {
    auto model = scriptedOsqp();
    model->every_solve = test::ScriptedModelSolve{ std::nullopt, nullptr, gap };
    auto prob = std::make_shared<test::ScriptedProb>(model);
    prob->createVariables({ "x0", "x1" }, { -1.0, -1.0 }, { 1.0, 1.0 });
    prob->addCost(std::make_shared<ChargedTargetCost>(prob->getVars(), DblVec{ 0.0, 0.0 }));
    BasicTrustRegionSQP solver(prob);
    solver.getParameters().min_approx_improve = 1e-12;  // reach the tiny box before a small-improvement exit
    solver.initialize({ 0.0, 0.0 });
    solver.optimize();
    EXPECT_EQ(solver.results().exit_reason, EXIT_TINY_TRUST_REGION);
    return solver.results().tiny_trust_region_after_uncertified;
  };
  EXPECT_FALSE(run(0.0));
  EXPECT_TRUE(run(1.0));
}

TEST_F(ScoTermination, UnconvergedSolveIsAProposalThatCannotExit)  // NOLINT
{
  auto model = scriptedOsqp();
  model->every_solve = test::ScriptedModelSolve{ CVX_UNCONVERGED, nullptr, std::nullopt };
  const OptResults r = runScripted(model);
  EXPECT_EQ(r.n_unconverged_qp_solves, model->solves);
  EXPECT_NE(r.exit_reason, EXIT_SMALL_IMPROVEMENT);
  // Neither small-improvement exit is taken; the solve ends on the tiny box, flagged
  EXPECT_EQ(r.exit_reason, EXIT_TINY_TRUST_REGION);
  EXPECT_TRUE(r.tiny_trust_region_after_uncertified);
  EXPECT_GT(r.n_suppressed_exits, 0);
  EXPECT_NEAR(r.x[0], 0.8, 1e-3);
}

TEST_F(ScoTermination, TinyBoxAfterFailureShrinkIsFlagged)  // NOLINT
{
  auto model = scriptedOsqp();
  model->script.push_back({ CVX_FAILED, nullptr, std::nullopt });
  const OptResults r = runScripted(model, [](BasicTrustRegionSQPParameters& p) { p.trust_box_size = 1.5e-4; });
  EXPECT_EQ(r.exit_reason, EXIT_TINY_TRUST_REGION);
  EXPECT_TRUE(r.tiny_trust_region_after_uncertified);
}
