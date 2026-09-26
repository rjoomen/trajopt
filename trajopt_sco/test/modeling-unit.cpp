#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_sco/expr_op_overloads.hpp>
#include <trajopt_sco/modeling.hpp>
#include <trajopt_sco/solver_interface.hpp>

using namespace sco;

/** @brief Builds a problem whose variables have the given bounds */
OptProb::Ptr problemWithBounds(const DblVec& lb, const DblVec& ub)
{
  auto prob = std::make_shared<OptProb>();
  std::vector<std::string> names;
  names.reserve(lb.size());
  for (std::size_t i = 0; i < lb.size(); ++i)
    names.push_back("x_" + std::to_string(i));
  prob->createVariables(names, lb, ub);
  return prob;
}

TEST(modeling, getClosestFeasiblePointClampsBothBounds)  // NOLINT
{
  const double delta = 1e-3;
  auto prob = problemWithBounds(DblVec{ -1, -1, -1, -1 }, DblVec{ 1, 1, 1, 1 });

  const DblVec y = prob->getClosestFeasiblePoint(DblVec{ 0.0, -2.0, 2.0, -1.0 }, delta);

  EXPECT_DOUBLE_EQ(y[0], 0.0);           // interior, untouched
  EXPECT_DOUBLE_EQ(y[1], -1.0 + delta);  // below the lower bound
  EXPECT_DOUBLE_EQ(y[2], 1.0 - delta);   // above the upper bound
  EXPECT_DOUBLE_EQ(y[3], -1.0 + delta);  // exactly on the lower bound
}

TEST(modeling, getClosestFeasiblePointInsetsFromBothBounds)  // NOLINT
{
  const double delta = 1e-3;
  auto prob = problemWithBounds(DblVec{ -1, -1 }, DblVec{ 1, 1 });

  // Inside the bounds but closer to them than delta
  const DblVec y = prob->getClosestFeasiblePoint(DblVec{ -1.0 + (delta / 2), 1.0 - (delta / 2) }, delta);

  EXPECT_DOUBLE_EQ(y[0], -1.0 + delta);
  EXPECT_DOUBLE_EQ(y[1], 1.0 - delta);
}

TEST(modeling, getClosestFeasiblePointRespectsNarrowBounds)  // NOLINT
{
  const double delta = 1e-3;
  // Bounds narrower than 2 * delta, and a zero width bound: there is no room to inset
  const DblVec lb{ 0.0, 0.5 };
  const DblVec ub{ delta, 0.5 };
  auto prob = problemWithBounds(lb, ub);

  for (const double x : { -1.0, 0.0, 0.25, 1.0 })
  {
    const DblVec y = prob->getClosestFeasiblePoint(DblVec{ x, x }, delta);
    for (std::size_t i = 0; i < y.size(); ++i)
    {
      EXPECT_GE(y[i], lb[i]);
      EXPECT_LE(y[i], ub[i]);
      EXPECT_DOUBLE_EQ(y[i], (lb[i] + ub[i]) / 2);  // midpoint is as far inside as the bounds allow
    }
  }
}

TEST(modeling, getClosestFeasiblePointLeavesUnboundedVarsAlone)  // NOLINT
{
  const double inf = std::numeric_limits<double>::infinity();
  auto prob = problemWithBounds(DblVec{ -inf, -inf }, DblVec{ inf, 1.0 });

  const DblVec y = prob->getClosestFeasiblePoint(DblVec{ -1e9, -1e9 });

  EXPECT_DOUBLE_EQ(y[0], -1e9);
  EXPECT_DOUBLE_EQ(y[1], -1e9);
}

TEST(modeling, getClosestFeasiblePointIsIdempotent)  // NOLINT
{
  const double delta = 1e-3;
  auto prob = problemWithBounds(DblVec{ -1, -1, 0.0 }, DblVec{ 1, 1, delta });

  const DblVec once = prob->getClosestFeasiblePoint(DblVec{ -5.0, 5.0, 5.0 }, delta);
  const DblVec twice = prob->getClosestFeasiblePoint(once, delta);

  EXPECT_EQ(once, twice);
}

TEST(ConvexObjective, ValueIgnoresSlackVariables)  // NOLINT
{
  const std::vector<ModelType> solvers = availableSolvers();
  if (solvers.empty())
    GTEST_SKIP() << "no convex solver available";
  const Model::Ptr model = createModel(solvers.front());
  const Var x = model->addVar("x");
  model->update();

  ConvexObjective objective(model.get());
  objective.addHinge(x - 1.0, 2.0);  // 2 * max(0, x - 1)
  objective.addAbs(x + 0.5, 3.0);    // 3 * |x + 0.5|
  model->update();

  // Arbitrary slack values: the hinge slack at 0, the absolute-value slacks at 7
  DblVec values(model->getVars().size(), 7.0);
  values[x.var_rep->index] = 2.0;
  values[objective.vars_[0].var_rep->index] = 0.0;
  EXPECT_NEAR(objective.value(values), (2.0 * 1.0) + (3.0 * 2.5), 1e-12);
}
