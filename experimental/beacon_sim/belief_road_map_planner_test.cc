
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
namespace {
MappedLandmarks create_mapped_landmarks() {
    const Eigen::Vector2d beacon_in_local{-7.5, -2.5};
    constexpr double POSITION_UNCERTAINTY_M = 0.1;
    const Eigen::Matrix2d cov_in_local{{{POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M, 0.0},
                                        {0.0, POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M}}};
    return MappedLandmarks{
        .beacon_ids = {123},
        .beacon_in_local = {beacon_in_local},
        .cov_in_local = cov_in_local,
    };
}

planning::RoadMap create_road_map() {
    constexpr double NODE_SPACING_M = 5.0;
    constexpr int NUM_ROWS = 3;
    constexpr int NUM_COLS = 3;
    constexpr int NUM_NODES = NUM_ROWS * NUM_COLS;

    std::vector<Eigen::Vector2d> points;
    Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(NUM_NODES, NUM_NODES);

    constexpr double ROW_OFFSET_M = -(NUM_ROWS - (NUM_ROWS % 2)) / 2.0 * NODE_SPACING_M;
    constexpr double COL_OFFSET_M = -(NUM_COLS - (NUM_COLS % 2)) / 2.0 * NODE_SPACING_M;

    for (int row = 0; row < NUM_ROWS; row++) {
        for (int col = 0; col < NUM_COLS; col++) {
            const double x_pos_m = NODE_SPACING_M * col + COL_OFFSET_M;
            const double y_pos_m = NODE_SPACING_M * row + ROW_OFFSET_M;
            points.push_back(Eigen::Vector2d{x_pos_m, y_pos_m});

            // Add edge to the right if it exists
            const int node_idx = col + row * NUM_COLS;
            if (col < (NUM_COLS - 1)) {
                const int neighbor_idx = (col + 1) + row * NUM_COLS;
                adj(node_idx, neighbor_idx) = 1.0;
                adj(neighbor_idx, node_idx) = 1.0;
            }
            // Add edge up if it exists
            if (row < (NUM_ROWS - 1)) {
                const int neighbor_idx = col + (row + 1) * NUM_COLS;
                adj(node_idx, neighbor_idx) = 1.0;
                adj(neighbor_idx, node_idx) = 1.0;
            }
        }
    }
    return planning::RoadMap{
        .points = std::move(points),
        .adj = std::move(adj),
    };
}

std::tuple<planning::RoadMap, EkfSlam> create_environment(const EkfSlamConfig &ekf_config) {
    // Create the environment depicted below
    //
    //
    //                            Start
    //                              │
    //                              │          +Y▲
    //                              │5m          │
    //                              │            └──►
    //                        5m    │   5m          +X
    //                    X─────────X─────────X
    //                   6│        7│        8│
    //                    │         │         │
    //                    │         │         │ 5m
    //                    │         │         │
    //                    │         │(0,0)    │
    //                    X─────────X─────────X
    //                   3│        4│        5│
    //                    │         │         │
    //                B   │         │         │ 5m
    //   (-7.5, -2.5)     │         │         │
    //                    │         │         │   5m
    //                    X─────────X─────────X───────── Goal
    //                    0         1         2
    //
    //
    //
    //
    // Note that:
    //  - each X is a PRM node,
    //  - B represents a beacon,
    //  - The world origin exists at the middle prm node,
    //  - The robot starts at (0, 10)
    //  - node indices start in the lower left and increase to the right, then increase up

    const auto mapped_landmarks = create_mapped_landmarks();
    const auto road_map = create_road_map();
    auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
    constexpr bool LOAD_OFF_DIAGONALS = true;
    // Move the robot to (0, 10) and have it face down
    const liegroups::SE2 old_robot_from_new_robot(-std::numbers::pi / 2.0, {0, 10});
    ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    return {road_map, ekf_slam};
}
}  // namespace

TEST(BeliefRoadMapPlannerTest, grid_road_map) {
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
    const auto &[road_map, ekf_slam] = create_environment(ekf_config);
    const Eigen::Vector2d GOAL_STATE = {10, -5};
    constexpr int NUM_START_CONNECTIONS = 1;
    constexpr int NUM_GOAL_CONENCTIONS = 1;
    constexpr double UNCERTAINTY_TOLERANCE = 1e-3;

    // Action
    const auto maybe_plan = compute_belief_road_map_plan(
        road_map, ekf_slam, {}, GOAL_STATE, MAX_SENSOR_RANGE_M, NUM_START_CONNECTIONS,
        NUM_GOAL_CONENCTIONS, UNCERTAINTY_TOLERANCE);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    std::cout << "Num Nodes: " << plan.nodes.size() << std::endl;
    for (int i = 0; i < static_cast<int>(plan.nodes.size()); i++) {
        std::cout << i << " idx: " << plan.nodes.at(i) << std::endl;
    }
}

TEST(BeliefRoadMapPlannerTest, compute_edge_transform_no_measurements) {
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
    const auto &[road_map, ekf_slam] = create_environment(ekf_config);

    constexpr int START_NODE_IDX = 6;
    constexpr int END_NODE_IDX = 3;
    const liegroups::SE2 local_from_robot =
        liegroups::SE2::trans(road_map.points.at(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.points.at(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = detail::compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TEST(BeliefRoadMapPlannerTest, compute_edge_transform_with_measurement) {
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
    const auto &[road_map, ekf_slam] = create_environment(ekf_config);

    constexpr int START_NODE_IDX = 3;
    constexpr int END_NODE_IDX = 0;
    const liegroups::SE2 local_from_robot =
        liegroups::SE2::trans(road_map.points.at(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.points.at(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = detail::compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

}  // namespace robot::experimental::beacon_sim
