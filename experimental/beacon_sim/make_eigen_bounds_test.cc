#include "experimental/beacon_sim/make_eigen_bounds.hh"

#include "Eigen/Core"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
bool eigen_debug = false;

TEST(MakeEigenBoundsTest, step_eigen_bound_no_information) {
    const double start_info_eigen_value = 0.5;
    // Action
    double newbound =
        step_lower_eigen_bound({.lower_eigen_value_information = start_info_eigen_value,
                                .upper_eigen_value_dynamics = 0.008,
                                .upper_eigen_value_measurement = 0.000001,
                                .lower_eigen_value_process_noise = 0.02});

    // Verification
    // lower bound should increase as more information is required travel the edge (as no info is
    // gained and process noise is added)
    ASSERT_GT(newbound, start_info_eigen_value);
}

TEST(MakeEigenBoundsTest, step_eigen_bound_information) {
    const double start_info_eigen_value = 0.5;
    // Action
    double newbound =
        step_lower_eigen_bound({.lower_eigen_value_information = start_info_eigen_value,
                                .upper_eigen_value_dynamics = 0.5,
                                .upper_eigen_value_measurement = 1.4,
                                .lower_eigen_value_process_noise = 0.008});

    // Verification
    // lower bound should decrease as information is added along the edge that outweighs the process
    // noise
    ASSERT_LT(newbound, start_info_eigen_value);
}

TEST(MakeEigenBoundsTest, sdp_bound) {
    drake::solvers::MathematicalProgram program;
    const Eigen::Matrix3d F{{1, 0, 0}, {0, 1, -1}, {0, 0, 1}};
    const Eigen::Matrix3d Q = Eigen::Matrix3d::Zero();
    const Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    const double lower_bound_at_end = 2000.0;

    auto lower_bound_at_start = program.NewContinuousVariables(1);
    auto cov_at_start = program.NewSymmetricContinuousVariables(3, "cov_at_start");
    program.AddPositiveSemidefiniteConstraint(cov_at_start);  // PSD on \Sigma_t-1
    program.AddPositiveSemidefiniteConstraint(lower_bound_at_start(0) * Eigen::Matrix3d::Identity() -
                                              cov_at_start);  // \Sigma_t-1 - \lambda I \succeq 0

    program.AddLinearConstraint(lower_bound_at_start(0) >= 0);
    program.AddPositiveSemidefiniteConstraint(
        Eigen::Matrix3d::Identity() - ((1/lower_bound_at_end * Eigen::Matrix3d::Identity() - M) *
                                          (F * cov_at_start * F.transpose() + Q)));

    program.AddLinearCost(-1.0 * lower_bound_at_start(0));

    std::cout << "program " << program << std::endl;

    auto solution = drake::solvers::Solve(program);
    std::cout << " solve succeeded " << solution.is_success() << std::endl;
    if (solution.is_success()) {
        auto result = solution.get_solution_result();
        std::cout << "result type" << result << std::endl;
        auto answer = solution.get_x_val();
        std::cout << "answer: " << answer << std::endl;


    }
}

TEST(MakeEigenBoundsTest, compute_edge_transform_no_measurements) {
    // Setup
    const EkfSlamConfig ekf_config{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100.0,
        .along_track_process_noise_m_per_rt_meter = 0.05,
        .cross_track_process_noise_m_per_rt_meter = 0.05,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-1,
        .bearing_measurement_noise_rad = 1e-1,
        .on_map_load_position_uncertainty_m = 2.0,
        .on_map_load_heading_uncertainty_rad = 0.5,
    };
    constexpr double MAX_SENSOR_RANGE_M = 3.0;
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);

    constexpr int START_NODE_IDX = 6;
    constexpr int END_NODE_IDX = 3;
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(END_NODE_IDX));

    const Eigen::Vector2d start_pos = road_map.point(START_NODE_IDX);
    // Action
    const double initial_info_min_eigen_value_bound = 123.0;
    eigen_debug = true;
    const auto edge_belief_transform = compute_backwards_eigen_bound_transform(
        initial_info_min_eigen_value_bound, local_from_robot, start_pos, ekf_slam.config(),
        ekf_slam.estimate(), {{}}, MAX_SENSOR_RANGE_M);

    // Verification
    EXPECT_LT(initial_info_min_eigen_value_bound, edge_belief_transform);
}

}  // namespace robot::experimental::beacon_sim
