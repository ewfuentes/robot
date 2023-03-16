
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/common/symbolic/expression.h"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(DrakeOptimizationTest, example) {
    // Setup
    drake::solvers::MathematicalProgram program;
    const auto x = program.NewContinuousVariables(1);
    program.AddQuadraticCost(x[0] * x[0]);
    program.AddLinearConstraint(x[0] >= 1.0);

    // Action
    const auto result = Solve(program);

    // Verification
    EXPECT_NEAR(result.GetSolution(x[0]), 1.0, 1e-6);
}
}  // namespace robot::experimental::beacon_sim
