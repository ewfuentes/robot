
#include "gtest/gtest.h"

#include <cmath>
#include <iostream>

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/clarabel_solver.h"
#include "drake/solvers/semidefinite_relaxation.h"

namespace robot::experimental::overhead_matching {
namespace {
template <int rows, int cols>
using VarMatrix = Eigen::Matrix<drake::symbolic::Expression, rows, cols>;
template <int rows>
using VarVector = Eigen::Vector<drake::symbolic::Expression, rows>;

drake::symbolic::Expression build_q_matrix(
	const drake::solvers::VectorXDecisionVariable &vars,
	const Eigen::Vector2d point_in_world,
	const double bearing_in_robot) {
	const auto &t_x = vars(0);
	const auto &t_y = vars(1);
	const auto &cos = vars(2);
	const auto &sin = vars(3);
	const VarMatrix<2, 2> robot_from_world_rot = (VarMatrix<2, 2>() << cos, -sin, sin, cos).finished();     
	const Eigen::Vector2<drake::symbolic::Expression> robot_from_world_trans{t_x, t_y};

	const Eigen::Vector2<drake::symbolic::Expression> point_in_robot = robot_from_world_rot * point_in_world +  robot_from_world_trans;

	std::cout << "bearing in robot: " << bearing_in_robot << std::endl;
	const Eigen::Vector2d u_in_robot{std::cos(bearing_in_robot), std::sin(bearing_in_robot)};

	std::cout << "u in robot: " << u_in_robot.transpose() << std::endl;

	const VarVector<2> error = point_in_robot - u_in_robot * (u_in_robot.transpose() * point_in_robot);
	const drake::symbolic::Expression out = (error.transpose() * error)(0);

	return out;
}
}

TEST(PoseOptimizerDrakeTest, points_and_bearings) {
    // Setup
    drake::solvers::MathematicalProgram prog;
    const auto x = prog.NewContinuousVariables(4);
    const auto &t_x = x(0);
    const auto &t_y = x(1);
    const auto &cos = x(2);
    const auto &sin = x(3);

    drake::symbolic::Expression exp = build_q_matrix(x, Eigen::Vector2d{0, -3}, std::atan2(-3, 2));
    exp += build_q_matrix(x, Eigen::Vector2d{0, 4}, std::atan2(4, 2));

    prog.AddCost(exp);
    prog.AddConstraint(sin * sin + cos * cos == 1);
    auto sdp_options = drake::solvers::SemidefiniteRelaxationOptions();
    sdp_options.set_to_strongest();
    const auto sdp_prog = drake::solvers::MakeSemidefiniteRelaxation(prog, sdp_options);

    const auto solver = drake::solvers::ClarabelSolver();
    const auto result = solver.Solve(*sdp_prog);

    // Action
    std::cout << "original program:" << std::endl;
    std::cout << prog << std::endl;
    std::cout << "sdp relaxed program:" << std::endl;
    std::cout << *sdp_prog << std::endl;
    std::cout << "is success: " << result.is_success() << std::endl;
    if (result.is_success()) {
	int idx = 0;
	const auto psd_constraint_vars = sdp_prog->positive_semidefinite_constraints().front().variables();
	std::cout << "psd constraint vars:" << psd_constraint_vars << std::endl;
	Eigen::MatrixXd psd_result = result.GetSolution(psd_constraint_vars).reshaped(5, 5);
	std::cout << psd_result << std::endl;
    }
    
    // Verification
}
}
