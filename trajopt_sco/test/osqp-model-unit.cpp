#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <cmath>
#include <gtest/gtest.h>
TRAJOPT_IGNORE_WARNINGS_POP

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
  auto config = std::make_shared<OSQPModelConfig>();
  config->settings.max_iter = 1;
  config->settings.polishing = 0;
  const Model::Ptr model = createModel(ModelType::OSQP, config);
  setupOneSidedProblem(*model);
  model->optimize();
  EXPECT_TRUE(std::isfinite(model->getDualityGap()));
}
