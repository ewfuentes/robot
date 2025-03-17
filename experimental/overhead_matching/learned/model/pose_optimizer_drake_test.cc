
#include <cmath>
#include <iostream>
#include <optional>
#include <regex>
#include <numbers>
#include <cstring>

#include "common/liegroups/se2.hh"
#include "common/matplotlib.hh"
#include "drake/solvers/clarabel_solver.h"
#include "drake/solvers/scs_solver.h"
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
                                const Eigen::Vector2d point_in_world, const double bearing_in_robot) {
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

drake::symbolic::Expression reprojection_error(const auto & vars,
                                               const Eigen::Vector2d &point_in_world, const Eigen::Vector2d &obs_point_in_robot) {

    const auto &t_x = vars(0);
    const auto &t_y = vars(1);
    const auto &cos = vars(2);
    const auto &sin = vars(3);

    const VarMatrix<2, 2> robot_from_world_rot =
        (VarMatrix<2, 2>() << cos, -sin, sin, cos).finished();
    const Eigen::Vector2<drake::symbolic::Expression> robot_from_world_trans{t_x, t_y};

    const Eigen::Vector2<drake::symbolic::Expression> point_in_robot =
        robot_from_world_rot * point_in_world + robot_from_world_trans;

    const VarVector<2> error = obs_point_in_robot - point_in_robot;
    return (error.transpose() * error)(0);

}
}  // namespace

TEST(PoseOptimizerDrakeTest, points_and_points_dual) {
    drake::solvers::MathematicalProgram prog;
    const auto indets = (
        Eigen::VectorX<drake::symbolic::Variable>(4) << 
            drake::symbolic::Variable("tx"),
            drake::symbolic::Variable("ty"),
            drake::symbolic::Variable("c"),
            drake::symbolic::Variable("s")
    ).finished();

    std::cout << indets << std::endl;

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
        const Eigen::Vector2d pt_in_robot = ego_from_world * pt_in_world;
        total_cost += reprojection_error(indets, pt_in_world, pt_in_robot);
    }
    const drake::symbolic::Polynomial cost_poly(total_cost, drake::symbolic::Variables(indets));

    std::cout << cost_poly << std::endl;
    auto monomial_map = cost_poly.monomial_to_coefficient_map();

    Eigen::MatrixXd Q(5, 5);
    for (int i = 0; i < 5; i++) {
        const auto mono_1 = i == 0 ? drake::symbolic::Monomial() : drake::symbolic::Monomial(indets(i - 1));
        for (int j = i; j < 5; j++) {
            const auto mono_2 = j == 0 ? drake::symbolic::Monomial() : drake::symbolic::Monomial(indets(j - 1));
            const double value = get_constant_value(monomial_map[mono_1 * mono_2]);
            std::cout << "term: " << mono_1 * mono_2 << " " << value << std::endl;
            const double factor = i == j ? 1.0 : 0.5;
            Q(i, j) = factor * value;
            Q(j, i) = factor * value;
        }
    }
    std::cout << Q << std::endl;

    Eigen::MatrixXd homogenizing = Eigen::MatrixXd::Zero(5, 5);
    homogenizing(0, 0) = 1.0;

    Eigen::MatrixXd det_is_one = Eigen::MatrixXd::Zero(5, 5);
    det_is_one(0, 0) = 1.0;
    det_is_one(3, 3) = -1.0;
    det_is_one(4, 4) = -1.0;

    const auto dual_vars = prog.NewContinuousVariables(2);
    // const auto H = Q - dual_vars[0] * homogenizing - dual_vars[1] * det_is_one;
    prog.AddLinearMatrixInequalityConstraint({Q, homogenizing, det_is_one}, dual_vars);
    // prog.AddPositiveSemidefiniteConstraint(-H);
    prog.AddCost(dual_vars[0]);

    // std::cout << "H: " << std::endl << H << std::endl;

    std::cout << "dual program: " << prog << std::endl;

    const auto solver = drake::solvers::ClarabelSolver();
    // const auto solver = drake::solvers::ScsSolver();
    drake::solvers::SolverOptions options;
    options.SetOption(drake::solvers::CommonSolverOption::kPrintToConsole, true);
    options.SetOption(drake::solvers::CommonSolverOption::kStandaloneReproductionFileName, "/tmp/scs_repro.py");
    const auto result = solver.Solve(prog, std::nullopt, options);

    std::cout << "is success: " << result.is_success() << std::endl;
    if (result.is_success()) {
        std::cout << "Optimal cost: " << result.get_optimal_cost() << std::endl;
        const Eigen::VectorXd dual_solution = result.GetDualSolution(prog.linear_matrix_inequality_constraints().front());
        std::cout << "Dual Solution: " << dual_solution.transpose() << std::endl;

        Eigen::MatrixXd psd_result(5, 5);
        int sol_idx = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = i; j < 5; j++) {
                psd_result(i, j) = dual_solution(sol_idx);
                psd_result(j, i) = dual_solution(sol_idx);
                sol_idx++;
            }
        }

        std::cout << psd_result << std::endl;

        const auto svd_result = psd_result.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Singular values: " << svd_result.singularValues().transpose() << std::endl;
        std::cout << "left singular vectors: " << std::endl << svd_result.matrixU() << std::endl;

        if (svd_result.singularValues()(1) < 0.1) {

            std::cout << "Found rank 1 solution! " << std::endl;
            Eigen::VectorXd solution = svd_result.matrixU().col(0).transpose();
            solution = solution / solution(0);

            std::cout << solution.transpose() << std::endl;
        }
    }
}

TEST(PoseOptimizerDrakeTest, points_and_points) {
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
        const Eigen::Vector2d pt_in_robot = ego_from_world * pt_in_world;
        total_cost += reprojection_error(x, pt_in_world, pt_in_robot);
    }

    prog.AddCost(total_cost);
    prog.AddConstraint(sin * sin + cos * cos == 1);
    auto sdp_options = drake::solvers::SemidefiniteRelaxationOptions();
    sdp_options.set_to_strongest();
    const auto sdp_prog = drake::solvers::MakeSemidefiniteRelaxation(prog, sdp_options);

    std::cout << "sdp relaxed program:" << std::endl;
    std::cout << *sdp_prog << std::endl;

    const auto solver = drake::solvers::ClarabelSolver();
    drake::solvers::SolverOptions options;
    options.SetOption(drake::solvers::CommonSolverOption::kPrintToConsole, true);
    const auto result = solver.Solve(*sdp_prog, std::nullopt, options);

    std::cout << "is success: " << result.is_success() << std::endl;
    if (result.is_success()) {
        std::cout << "Optimal cost: " << result.get_optimal_cost() << std::endl;
        const auto psd_constraint_vars =
            sdp_prog->positive_semidefinite_constraints().front().variables();
        std::cout << "psd constraint vars:" << std::endl
                  << psd_constraint_vars.reshaped(5, 5) << std::endl;
        Eigen::MatrixXd psd_result = result.GetSolution(psd_constraint_vars).reshaped(5, 5);
        std::cout << psd_result << std::endl;

        const auto svd_result = psd_result.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Singular values: " << svd_result.singularValues().transpose() << std::endl;
        std::cout << "left singular vectors: " << std::endl << svd_result.matrixU() << std::endl;

        if (svd_result.singularValues()(1) < 0.1) {

            std::cout << "Found rank 1 solution! " << std::endl;
            Eigen::VectorXd solution = svd_result.matrixU().col(0).transpose();
            solution = solution / solution(4);

            std::cout << solution.transpose() << std::endl;
        }
       // else {
       //     const Eigen::VectorXd expected_vector = (Eigen::VectorXd(5) << 2.0, 0.0, 1.0, 0.0, 1.0).finished();
       //     const Eigen::MatrixXd expected_solution = expected_vector * expected_vector.transpose();

       //     std::cout << "expected solution: " << std::endl << expected_solution << std::endl;

       //     std::cout << "expected solution trace: " << expected_solution.trace() << std::endl;
       //     const Eigen::MatrixXd normed_expected_solution = expected_solution / expected_solution.trace();
       //     const double projection_along_expected = (psd_result * normed_expected_solution).trace();
       //     std::cout << "projection_along_expected: " << projection_along_expected << std::endl;

       //     const Eigen::MatrixXd remainder = psd_result - projection_along_expected * normed_expected_solution;

       //     std::cout << "remainder: " << std::endl << remainder << std::endl;
       //     const auto remainder_svd = remainder.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
       //     std::cout << "remainder singular values: " << remainder_svd.singularValues().transpose() << std::endl;
       //     std::cout << "remainder singular vectors: " << std::endl << remainder_svd.matrixU() << std::endl;

       // }
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

    const std::vector<Eigen::Vector2d> pts_in_world = {
        Eigen::Vector2d{0, -3},
        Eigen::Vector2d{0, 4},
        Eigen::Vector2d{5, 0},
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

    prog.AddLinearConstraint(sin, -1.0, 1.0);
    prog.AddLinearConstraint(cos, -1.0, 1.0);

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
    // std::cout << "original program:" << std::endl;
    // std::cout << prog << std::endl;
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

        const auto svd_result = psd_result.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "Singular values: " << svd_result.singularValues().transpose() << std::endl;
        std::cout << "left singular vectors: " << std::endl << svd_result.matrixU() << std::endl;

        if (svd_result.singularValues()(1) < 0.1) {

            std::cout << "Found rank 1 solution! " << std::endl;
            Eigen::VectorXd solution = svd_result.matrixU().col(0).transpose();
            solution = solution / solution(4);

            std::cout << solution.transpose() << std::endl;
        }
       // else {
       //     const Eigen::VectorXd expected_vector = (Eigen::VectorXd(5) << 2.0, 0.0, 1.0, 0.0, 1.0).finished();
       //     const Eigen::MatrixXd expected_solution = expected_vector * expected_vector.transpose();

       //     std::cout << "expected solution: " << std::endl << expected_solution << std::endl;

       //     std::cout << "expected solution trace: " << expected_solution.trace() << std::endl;
       //     const Eigen::MatrixXd normed_expected_solution = expected_solution / expected_solution.trace();
       //     const double projection_along_expected = (psd_result * normed_expected_solution).trace();
       //     std::cout << "projection_along_expected: " << projection_along_expected << std::endl;

       //     const Eigen::MatrixXd remainder = psd_result - projection_along_expected * normed_expected_solution;

       //     std::cout << "remainder: " << std::endl << remainder << std::endl;
       //     const auto remainder_svd = remainder.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
       //     std::cout << "remainder singular values: " << remainder_svd.singularValues().transpose() << std::endl;
       //     std::cout << "remainder singular vectors: " << std::endl << remainder_svd.matrixU() << std::endl;

       // }
    }

    // Verification
}
}  // namespace robot::experimental::overhead_matching
