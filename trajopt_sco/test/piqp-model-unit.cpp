#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
TRAJOPT_IGNORE_WARNINGS_POP

#include <tesseract/common/logging.h>
#include <trajopt_sco/expr_op_overloads.hpp>
#include <trajopt_sco/expr_ops.hpp>
#include <trajopt_sco/piqp_interface.hpp>

using namespace sco;

TEST(PIQPModel, StatusClassification)  // NOLINT
{
  EXPECT_EQ(piqpStatusToCvxOptStatus(piqp::Status::PIQP_SOLVED), CVX_SOLVED);
  EXPECT_EQ(piqpStatusToCvxOptStatus(piqp::Status::PIQP_MAX_ITER_REACHED), CVX_UNCONVERGED);
  EXPECT_EQ(piqpStatusToCvxOptStatus(piqp::Status::PIQP_PRIMAL_INFEASIBLE), CVX_INFEASIBLE);
  EXPECT_EQ(piqpStatusToCvxOptStatus(piqp::Status::PIQP_DUAL_INFEASIBLE), CVX_INFEASIBLE);
  EXPECT_EQ(piqpStatusToCvxOptStatus(piqp::Status::PIQP_NUMERICS), CVX_FAILED);
}

// min (x - 1)^2 + (y - 2)^2 + z^2  s.t.  x + y = 1,  y <= 0.25,  z in [0.5, 0.5]  =>  x = 0.75, y = 0.25, z = 0.5
TEST(PIQPModel, EqualityInequalityAndPinnedVariable)  // NOLINT
{
  const Model::Ptr model = createModel(ModelType::PIQP);
  const Var x = model->addVar("x");
  const Var y = model->addVar("y");
  const Var z = model->addVar("z");
  model->update();

  QuadExpr objective = exprSquare(x - 1.0);
  exprInc(objective, exprSquare(y - 2.0));
  exprInc(objective, exprSquare(z));
  model->setObjective(objective);
  model->addEqCnt(x + y - 1.0, "sum");
  model->addIneqCnt(y - 0.25, "y_max");
  model->setVarBounds(z, 0.5, 0.5);
  model->update();

  ASSERT_EQ(model->optimize(), CVX_SOLVED);
  const DblVec values = model->getVarValues({ x, y, z });
  EXPECT_NEAR(values[0], 0.75, 1e-6);
  EXPECT_NEAR(values[1], 0.25, 1e-6);
  EXPECT_NEAR(values[2], 0.5, 1e-9);
}

TEST(PIQPModel, Infeasible)  // NOLINT
{
  const Model::Ptr model = createModel(ModelType::PIQP);
  const Var x = model->addVar("x");
  model->update();

  model->setObjective(exprSquare(x));
  model->addIneqCnt(AffExpr(x), "x_max");
  model->addIneqCnt(AffExpr(1.0) - x, "x_min");
  model->update();

  EXPECT_EQ(model->optimize(), CVX_INFEASIBLE);
}

TEST(PIQPModel, DualityGapInfiniteAfterInfeasibleSolve)  // NOLINT
{
  // x <= 0 and x >= 1: PIQP itself reports the infeasibility rather than a setup-time rejection
  const Model::Ptr model = createModel(ModelType::PIQP);
  const Var x = model->addVar("x");
  model->update();

  model->setObjective(exprSquare(x));
  model->addIneqCnt(AffExpr(x), "x_max");
  model->addIneqCnt(AffExpr(1.0) - x, "x_min");
  model->update();

  ASSERT_EQ(model->optimize(), CVX_INFEASIBLE);
  EXPECT_EQ(model->getDualityGap(), std::numeric_limits<double>::infinity());
}

TEST(PIQPModel, IterationCapReturnsUnconvergedValues)  // NOLINT
{
  // Silence the unconverged-solve WARN that every capped solve logs
  tesseract::common::getLogger()->set_level(spdlog::level::err);
  auto config = std::make_shared<PIQPModelConfig>();
  config->settings.max_iter = 1;
  const Model::Ptr model = createModel(ModelType::PIQP, config);
  const Var x = model->addVar("x");
  model->update();
  model->setObjective(exprSquare(x - 1.0));
  model->addIneqCnt(x - 0.5, "x_max");
  model->update();
  EXPECT_EQ(model->optimize(), CVX_UNCONVERGED);
  EXPECT_TRUE(std::isfinite(model->getVarValue(x)));
}

TEST(PIQPModel, ConfigSettingsAreUsed)  // NOLINT
{
  auto config = std::make_shared<PIQPModelConfig>();
  config->settings.kkt_solver = piqp::KKTSolver::dense_cholesky;
  const Model::Ptr model = createModel(ModelType::PIQP, config);
  const Var x = model->addVar("x");
  model->update();
  model->setObjective(exprSquare(x));

  EXPECT_EQ(model->optimize(), CVX_FAILED);
}

TEST(PIQPModel, DualityGapReportedWithGapCheckOff)  // NOLINT
{
  auto config = std::make_shared<PIQPModelConfig>();
  config->settings.check_duality_gap = false;
  const Model::Ptr model = createModel(ModelType::PIQP, config);
  const Var x = model->addVar("x");
  model->update();
  model->setObjective(exprSquare(x - 1.0));
  model->addIneqCnt(x - 0.5, "x_max");
  model->update();
  ASSERT_EQ(model->optimize(), CVX_SOLVED);
  EXPECT_TRUE(std::isfinite(model->getDualityGap()));
  EXPECT_LT(model->getDualityGap(), 1e-3);
}
