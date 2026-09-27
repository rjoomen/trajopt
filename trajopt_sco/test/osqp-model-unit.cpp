#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
TRAJOPT_IGNORE_WARNINGS_POP

#include <tesseract/common/logging.h>
#include <trajopt_sco/expr_op_overloads.hpp>
#include <trajopt_sco/expr_ops.hpp>
#include <trajopt_sco/osqp_interface.hpp>

using namespace sco;

namespace
{
/** @brief Minimize (x - 1)^2 subject to x <= 0.5 */
Var setupOneSidedProblem(Model& model)
{
  const Var x = model.addVar("x");
  model.update();
  model.setObjective(exprSquare(x - 1.0));
  model.addIneqCnt(x - 0.5, "x_max");
  model.update();
  return x;
}
}  // namespace

TEST(OSQPModel, StatusClassification)  // NOLINT
{
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_SOLVED), CVX_SOLVED);
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_SOLVED_INACCURATE), CVX_UNCONVERGED);
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_MAX_ITER_REACHED), CVX_UNCONVERGED);
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_PRIMAL_INFEASIBLE), CVX_INFEASIBLE);
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_DUAL_INFEASIBLE_INACCURATE), CVX_INFEASIBLE);
  EXPECT_EQ(osqpStatusToCvxOptStatus(OSQP_TIME_LIMIT_REACHED), CVX_FAILED);
}

TEST(OSQPModel, DualityGapSmallAtSolution)  // NOLINT
{
  const Model::Ptr model = createModel(ModelType::OSQP);
  setupOneSidedProblem(*model);
  ASSERT_EQ(model->optimize(), CVX_SOLVED);
  EXPECT_TRUE(std::isfinite(model->getDualityGap()));
  EXPECT_LT(model->getDualityGap(), 1e-4);
}

TEST(OSQPModel, DualityGapFiniteAtIterationCap)  // NOLINT
{
  // Silence the unconverged-solve WARN that every capped solve logs
  tesseract::common::getLogger()->set_level(spdlog::level::err);
  auto config = std::make_shared<OSQPModelConfig>();
  config->settings.max_iter = 1;
  config->settings.polishing = 0;
  const Model::Ptr model = createModel(ModelType::OSQP, config);
  setupOneSidedProblem(*model);
  model->optimize();
  EXPECT_TRUE(std::isfinite(model->getDualityGap()));
}

TEST(OSQPModel, IterationCapReturnsUnconvergedValues)  // NOLINT
{
  // Silence the unconverged-solve WARN that every capped solve logs
  tesseract::common::getLogger()->set_level(spdlog::level::err);
  auto config = std::make_shared<OSQPModelConfig>();
  config->settings.max_iter = 1;
  config->settings.polishing = 0;
  const Model::Ptr model = createModel(ModelType::OSQP, config);
  const Var x = setupOneSidedProblem(*model);
  EXPECT_EQ(model->optimize(), CVX_UNCONVERGED);
  EXPECT_TRUE(std::isfinite(model->getVarValue(x)));
}

TEST(OSQPModel, DualityGapInfiniteAfterInfeasibleSolve)  // NOLINT
{
  // x0 + x1 <= 1 solved first (feasible), then x0 + x1 >= 3 added as a second row: each row is valid on its
  // own, but together infeasible, so OSQP detects it during the solve rather than rejecting a bound at setup
  const Model::Ptr model = createModel(ModelType::OSQP);
  const Var x0 = model->addVar("x0");
  const Var x1 = model->addVar("x1");
  model->update();
  model->setObjective(exprSquare(x0) + exprSquare(x1));
  model->addIneqCnt(x0 + x1 - 1.0, "upper");
  model->update();
  ASSERT_EQ(model->optimize(), CVX_SOLVED);
  ASSERT_TRUE(std::isfinite(model->getDualityGap()));

  // Reuse the same model so a stale finite gap left over from the feasible solve would show
  model->addIneqCnt(AffExpr(3.0) - x0 - x1, "lower");
  model->update();
  EXPECT_EQ(model->optimize(), CVX_INFEASIBLE);
  EXPECT_EQ(model->getDualityGap(), std::numeric_limits<double>::infinity());
}
