
#include "experimental/beacon_sim/test_helpers.hh"

#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/precision_matrix_potential.hh"

namespace robot::experimental::beacon_sim {

MappedLandmarks create_grid_mapped_landmarks() {
    const Eigen::Vector2d beacon_in_local{-7.5, -2.5};
    constexpr double POSITION_UNCERTAINTY_M = 0.1;
    const Eigen::Matrix2d cov_in_local{{{POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M, 0.0},
                                        {0.0, POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M}}};
    return MappedLandmarks{
        .beacon_ids = {GRID_BEACON_ID},
        .beacon_in_local = {beacon_in_local},
        .cov_in_local = cov_in_local,
    };
}

planning::RoadMap create_grid_road_map() {
    const Eigen::Vector2d START_STATE{0.0, 10.0};
    const Eigen::Vector2d GOAL_STATE{10.0, -5.0};
    const double CONNECTION_RADIUS_M = 6.0;
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
    return planning::RoadMap(std::move(points), std::move(adj),
                             {{
                                 .start = START_STATE,
                                 .goal = GOAL_STATE,
                                 .connection_radius_m = CONNECTION_RADIUS_M,
                             }});
}

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_grid_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon) {
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

    const auto mapped_landmarks = create_grid_mapped_landmarks();
    const auto road_map = create_grid_road_map();
    auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
    constexpr bool LOAD_OFF_DIAGONALS = true;

    // Lone beacon potential
    const double lone_log_norm = -std::log(1 - p_lone_beacon);
    const double lone_param = std::log(p_lone_beacon) + lone_log_norm;
    const auto lone_potential =
        PrecisionMatrixPotential{.precision = Eigen::Matrix<double, 1, 1>{lone_param},
                                 .log_normalizer = lone_log_norm,
                                 .members = {GRID_BEACON_ID}};

    // Move the robot to (0, 10) and have it face down
    const liegroups::SE2 old_robot_from_new_robot(-std::numbers::pi / 2.0, {0, 10});
    ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    return {road_map, ekf_slam, lone_potential};
}

MappedLandmarks create_diamond_mapped_landmarks() {
    constexpr int LONE_BEACON_ID = 123;
    const Eigen::Vector2d LONE_BEACON_IN_LOCAL{-1, 6};
    constexpr int START_STACKED_BEACON_ID = 10;
    const Eigen::Vector2d STACKED_BEACON_IN_LOCAL{6, -1};
    constexpr int NUM_STACKED_BEACONS = 10;
    constexpr double POSITION_UNCERTAINTY_M = 0.1;
    constexpr int NUM_BEACONS = NUM_STACKED_BEACONS + 1;
    const Eigen::Matrix2d cov_in_local{{{POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M, 0.0},
                                        {0.0, POSITION_UNCERTAINTY_M * POSITION_UNCERTAINTY_M}}};

    MappedLandmarks out;
    out.cov_in_local = Eigen::MatrixXd::Zero(2 * NUM_BEACONS, 2 * NUM_BEACONS);
    // Add the lone beacon
    out.beacon_ids.push_back(LONE_BEACON_ID);
    out.beacon_in_local.push_back(LONE_BEACON_IN_LOCAL);
    out.cov_in_local.block<2, 2>(0, 0) = cov_in_local;

    // Add the stacked beacons
    for (int i = 1; i <= NUM_STACKED_BEACONS; i++) {
        out.beacon_ids.push_back(START_STACKED_BEACON_ID + i);
        out.beacon_in_local.push_back(STACKED_BEACON_IN_LOCAL);
        out.cov_in_local.block<2, 2>(2 * i, 2 * i) = cov_in_local;
    }

    return out;
}

planning::RoadMap create_diamond_road_map() {
    return planning::RoadMap(
        {{0.0, 0.0}, {0.0, 5.0}, {5.0, 0.0}, {5.0, 5.0}},
        (Eigen::MatrixXd(4, 4) << 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0).finished(),
        {{.start = {-2.0, 0.0}, .goal = {7.0, 5.0}, .connection_radius_m = 3.0}});
}

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_diamond_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon, const double p_no_stack_beacon,
    const double p_stacked_beacon) {
    // Create the environment depicted below
    //
    //            B (-1, 6)
    //                        3
    //              X────────X─────Goal
    //              │1       │  2m
    //              │        │
    //              │        │ 5m
    //              │        │
    //    Start─────X────────X 2
    //          2m   0
    //                  5m     Bx10
    //                         (6, -1)
    // Note that:
    // - each X is a PRM node,
    // - there is a beacon at (-1, 6) and there are 10 beacons at (6, -1)
    // - The robot starts at (-2, 0)

    const auto mapped_landmarks = create_diamond_mapped_landmarks();
    const auto road_map = create_diamond_road_map();
    auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
    constexpr bool LOAD_OFF_DIAGONALS = true;

    // Lone beacon potential
    const double lone_log_norm = -std::log(1 - p_lone_beacon);
    const double lone_param = std::log(p_lone_beacon) + lone_log_norm;
    const auto lone_potential =
        PrecisionMatrixPotential{.precision = Eigen::Matrix<double, 1, 1>{lone_param},
                                 .log_normalizer = lone_log_norm,
                                 .members = {123}};

    // Stacked Potential
    const auto stacked_potential = create_correlated_beacons({
        .p_beacon = p_stacked_beacon,
        .p_no_beacons = p_no_stack_beacon,
        .members = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    });
    const auto beacon_potential = lone_potential * stacked_potential;

    // Move the robot to (-2, 0) and have it face down
    const liegroups::SE2 old_robot_from_new_robot(-std::numbers::pi / 2.0, {-2, 0});
    ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    return {road_map, ekf_slam, beacon_potential};
}

}  // namespace robot::experimental::beacon_sim
