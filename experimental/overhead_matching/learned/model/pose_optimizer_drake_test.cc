
#include <cmath>
#include <iostream>
#include <regex>
#include <numbers>
#include <cstring>

#include "common/liegroups/se2.hh"
#include "common/matplotlib.hh"
#include "drake/solvers/clarabel_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/semidefinite_relaxation.h"
#include "gtest/gtest.h"
#include "fmt/base.h"
#include "fmt/format.h"
#include "Eigen/Geometry"

double crossProduct2D(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

namespace robot::experimental::overhead_matching {
namespace {
template <int rows, int cols>
using VarMatrix = Eigen::Matrix<drake::symbolic::Expression, rows, cols>;
template <int rows>
using VarVector = Eigen::Vector<drake::symbolic::Expression, rows>;

struct ProblemComponent {
    drake::symbolic::Expression cost;
    drake::symbolic::Expression dot_product;
};

void SetRelaxationInitialGuess(const Eigen::Ref<const Eigen::VectorXd> &y_expected,
                               drake::solvers::MathematicalProgram *relaxation) {
    const int N = y_expected.size() + 1;
    Eigen::MatrixX<drake::symbolic::Variable> X =
        Eigen::Map<const Eigen::MatrixX<drake::symbolic::Variable>>(
            relaxation->positive_semidefinite_constraints()[0].variables().data(), N, N);
    Eigen::VectorXd x_expected(N);
    x_expected << y_expected, 1;
    const Eigen::MatrixXd X_expected = x_expected * x_expected.transpose();
    relaxation->SetInitialGuess(X, X_expected);
}

ProblemComponent build_q_matrix(const drake::solvers::VectorXDecisionVariable &vars,
                                const Eigen::Vector2d point_in_world,
                                const double bearing_in_robot) {
    const auto &t_x = vars(0);
    const auto &t_y = vars(1);
    const auto &cos = vars(2);
    const auto &sin = vars(3);
    const VarMatrix<2, 2> robot_from_world_rot =
        (VarMatrix<2, 2>() << cos, -sin, sin, cos).finished();
    const Eigen::Vector2<drake::symbolic::Expression> robot_from_world_trans{t_x, t_y};

    const Eigen::Vector2<drake::symbolic::Expression> point_in_robot =
        robot_from_world_rot * point_in_world + robot_from_world_trans;

    const Eigen::Vector2d u_in_robot{std::cos(bearing_in_robot), std::sin(bearing_in_robot)};

    const drake::symbolic::Expression dot_product = u_in_robot.transpose() * point_in_robot;
    const VarVector<2> error = point_in_robot - u_in_robot * dot_product;
    const drake::symbolic::Expression cost = (error.transpose() * error)(0);

    return {
        .cost = cost,
        .dot_product = dot_product,
    };
}
}  // namespace

TEST(PoseOptimizerDrakeTest, points_and_bearings) {
    // Setup
    drake::solvers::MathematicalProgram prog;
    const auto x = prog.NewContinuousVariables(4);
    const auto &t_x = x(0);
    const auto &t_y = x(1);
    const auto &cos = x(2);
    const auto &sin = x(3);

    const std::vector<Eigen::Vector2d> pts_in_world = {
        Eigen::Vector2d{0, -3},
        Eigen::Vector2d{0, 4},
    };

    constexpr double ego_from_world_rot = 0.0;
    const Eigen::Vector2d ego_in_world{-2.0, 0.0};
    const liegroups::SE2 ego_from_world =
        liegroups::SE2(ego_from_world_rot, ego_in_world).inverse();

    drake::symbolic::Expression total_cost;
    for (const auto &pt_in_world : pts_in_world) {
        const Eigen::Vector2d pt_in_ego = ego_from_world * pt_in_world;
        const double bearing_rad = std::atan2(pt_in_ego.y(), pt_in_ego.x());
        const auto &[cost_comp, dot_prod] = build_q_matrix(x, pt_in_world, bearing_rad);
        total_cost += cost_comp;
        prog.AddConstraint(dot_prod >= 0);
    }

    prog.AddCost(total_cost);
    prog.AddConstraint(sin * sin + cos * cos == 1);
    auto sdp_options = drake::solvers::SemidefiniteRelaxationOptions();
    sdp_options.set_to_strongest();
    const auto sdp_prog = drake::solvers::MakeSemidefiniteRelaxation(prog, sdp_options);

    SetRelaxationInitialGuess(Eigen::Vector4d{2.0, 0, 1.0, 0.0}, &*sdp_prog);

    for (const auto &constraint : sdp_prog->GetAllConstraints()) {
        const bool is_satisfied = sdp_prog->CheckSatisfiedAtInitialGuess(constraint);
        std::cout << "is satisfied: " << is_satisfied << "\t" << constraint << std::endl;
    }
    for (const auto &cost : sdp_prog->GetAllCosts()) {
        std::cout << "cost value: " << sdp_prog->EvalBindingAtInitialGuess(cost) << "\t" << cost
                  << std::endl;
    }

    const auto solver = drake::solvers::ClarabelSolver();
    const auto result = solver.Solve(*sdp_prog);

    // Action
    std::cout << "original program:" << std::endl;
    std::cout << prog << std::endl;
    std::cout << "sdp relaxed program:" << std::endl;
    std::cout << *sdp_prog << std::endl;
    std::cout << "is success: " << result.is_success() << std::endl;
    if (result.is_success()) {
        std::cout << "Optimal cost: " << result.get_optimal_cost() << std::endl;
        const auto psd_constraint_vars =
            sdp_prog->positive_semidefinite_constraints().front().variables();
        std::cout << "psd constraint vars:" << std::endl
                  << psd_constraint_vars.reshaped(5, 5) << std::endl;
        Eigen::MatrixXd psd_result = result.GetSolution(psd_constraint_vars).reshaped(5, 5);
        std::cout << psd_result << std::endl;

        std::cout << "Eigenvalues: " << psd_result.eigenvalues().transpose() << std::endl;

        const auto svd_result = psd_result.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Singular values: " << svd_result.singularValues() << std::endl;
        std::cout << "left singular vectors: " << std::endl << svd_result.matrixU() << std::endl;
        std::cout << "right singular vectors: " << std::endl << svd_result.matrixV() << std::endl;
    }


    const double length_a = (pts_in_world[0] - pts_in_world[1]).norm();
    const Eigen::Vector2d a_in_robot = (ego_from_world * pts_in_world[0]).normalized();
    const Eigen::Vector2d b_in_robot = (ego_from_world * pts_in_world[1]).normalized();
    const double sin_alpha = std::abs(crossProduct2D(a_in_robot, b_in_robot));
    const double alpha = std::asin(sin_alpha);

    const double ratio = sin_alpha / length_a;

    const Eigen::VectorXd b_length = Eigen::VectorXd::LinSpaced(100, 0, length_a);
    const Eigen::VectorXd sin_beta = ratio * b_length;
    const Eigen::VectorXd beta = Eigen::asin(sin_beta.array());
    const Eigen::VectorXd gamma = -std::numbers::pi / 2.0 - (std::numbers::pi - alpha - beta.array());

    const Eigen::ArrayXd cos_gamma = Eigen::cos(gamma.array());
    const Eigen::ArrayXd sin_gamma = Eigen::sin(gamma.array());
    const Eigen::VectorXd x_locs = pts_in_world[1].x() + (b_length.array() * cos_gamma).array();
    const Eigen::VectorXd y_locs = pts_in_world[1].y() + (b_length.array() * sin_gamma).array();

    plot<Eigen::VectorXd>({PlotSignal{
        .x = x_locs,
        .y = y_locs,
    }});





    std::cout << "a in robot: " << a_in_robot.transpose() << " b in robot: " << b_in_robot.transpose() << std::endl;
    fmt::println("a: {} sin_alpha: {}", length_a, alpha);

     if (false) {
        const Eigen::VectorXd Xs = Eigen::VectorXd::LinSpaced(601, -6.0, 6.0);
        const Eigen::VectorXd Ys = Eigen::VectorXd::LinSpaced(601, -6.0, 6.0);
        const Eigen::VectorXd Ts = Eigen::VectorXd::LinSpaced(601, -std::numbers::pi, std::numbers::pi);
        Eigen::MatrixXd data(Xs.rows(), Ys.rows());
        drake::symbolic::Environment env;

        const int target_pt_idx = 0;

        for (int i = 0; i < Ys.rows(); i++) {
            for (int j = 0; j < Xs.rows(); j++) {
                env[t_x] = Xs(j);
                env[t_y] = Ys(i);

                const Eigen::Vector2d pt_in_world = pts_in_world.at(target_pt_idx);
                const Eigen::Vector2d pt_in_robot = ego_from_world * pt_in_world; 
                const double observed_bearing_rad = std::atan2(pt_in_robot.y(), pt_in_robot.x());
                const liegroups::SE2 test_ego_from_world(0.0, Eigen::Vector2d{Xs(j), Ys(i)});
                const Eigen::Vector2d test_pt_in_robot = test_ego_from_world * pt_in_world;
                const double bearing_in_unrotated_rad = std::atan2(test_pt_in_robot.y(), test_pt_in_robot.x());
                const double delta_bearing_rad = bearing_in_unrotated_rad - observed_bearing_rad;

                env[cos] = std::cos(delta_bearing_rad);
                env[sin] = std::sin(delta_bearing_rad);
                const double value = total_cost.Evaluate(env);
                data(i, j) = std::log10(value);

                if ((i == 0 && j == 0) || (Xs(j) == 2.0 && Ys(i) == 0.0)) {
                    fmt::println("x: {} y: {} pt_in_world: ({}, {}) pt_in_robot: ({}, {}) observed_bearing_rad: {} "
                                 "unrotated bearing: {} delta_bearing: {} value: {}", 
                                 Xs(j), Ys(i),
                                 pt_in_world.x(), pt_in_world.y(),
                                 pt_in_robot.x(), pt_in_robot.y(),
                                 observed_bearing_rad, bearing_in_unrotated_rad,
                                 delta_bearing_rad, value);
                }
            }
        }
        std::cout << std::endl;
        contourf(Xs, Ys, data, true);
     }
    // Verification
}
}  // namespace robot::experimental::overhead_matching
