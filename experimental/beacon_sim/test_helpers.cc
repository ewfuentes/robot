
#include "experimental/beacon_sim/test_helpers.hh"
#include <numbers>
#include "experimental/beacon_sim/anticorrelated_beacon_potential.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/precision_matrix_potential.hh"
#include "planning/road_map.hh"

namespace robot::experimental::beacon_sim {

MappedLandmarks create_grid_mapped_landmarks(const std::vector<Eigen::Vector2d> &beacon_locs, const std::vector<int> &beacon_IDs) {
    constexpr double POSITION_UNCERTAINTY_M = 0.1;
    const int cove_size = beacon_locs.size()*2;
    const Eigen::MatrixXd cov_in_local = Eigen::MatrixXd::Identity(cove_size,cove_size)*POSITION_UNCERTAINTY_M*POSITION_UNCERTAINTY_M;
    return MappedLandmarks{
        .beacon_ids = beacon_IDs,
        .beacon_in_local = beacon_locs,
        .cov_in_local = cov_in_local,
    };
}
planning::RoadMap create_grid_road_map(const Eigen::Vector2d start,const Eigen::Vector2d goal,const int NUM_ROWS,const int NUM_COLS) {
    const double CONNECTION_RADIUS_M = 6.0;
    constexpr double NODE_SPACING_M = 5.0;
    const int NUM_NODES = NUM_ROWS * NUM_COLS;

    std::vector<Eigen::Vector2d> points;
    Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(NUM_NODES, NUM_NODES);

    const double ROW_OFFSET_M = -(NUM_ROWS - (NUM_ROWS % 2)) / 2.0 * NODE_SPACING_M;
    const double COL_OFFSET_M = -(NUM_COLS - (NUM_COLS % 2)) / 2.0 * NODE_SPACING_M;

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
                                 .start = start,
                                 .goal = goal,
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

    const auto mapped_landmarks = create_grid_mapped_landmarks({{7.5,2.5}},{123});
    const auto road_map = create_grid_road_map({0,10},{10,-5},3,3);
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

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_stress_test_environment(
    const EkfSlamConfig &ekf_config) {
    // Create RoadMap
    //
    //
    //           A 3,4
    //
    //           0 3,3
    //          ╱│╲
    //         ╱ │ ╲
    //        ╱  │  ╲
    //       ╱   │   ╲
    //  0,0 S────1────G  6,0
    //       ╲   │   ╱
    //        ╲  │  ╱
    //         ╲ │ ╱
    //          ╲│╱
    //           2 3,-3
    //
    //           B 3,-4

    const planning::RoadMap road_map(
        std::vector<Eigen::Vector2d>{{3, 3}, {3, 0}, {3, -3}},
        (Eigen::Matrix3d() << 0, 1, 0, 1, 0, 1, 0, 1, 0).finished(),
        planning::StartGoalPair{.start = {0, 0}, .goal = {6, 0}, .connection_radius_m = 4.5});

    // Create EkfSlam
    const std::vector<int> BEACON_IDS{123, 456};
    EkfSlam ekf_slam(ekf_config, time::RobotTimestamp());
    const MappedLandmarks mapped_landmarks = {
        .beacon_ids = BEACON_IDS,
        .beacon_in_local = {{3, -4}, {3, 4}},
        .cov_in_local = Eigen::Matrix4d::Identity() * 1e-12,
    };
    const bool LOAD_OFF_DIAGONALS = true;
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);
    const liegroups::SE2 world_from_robot(-std::numbers::pi / 2.0, {0, 0});
    ekf_slam.estimate().local_from_robot(world_from_robot);

    // Create BeaconPotential
    const BeaconClique clique = {
        .p_beacon = 0.5,
        .p_no_beacons = 1e-12,
        .members = BEACON_IDS,
    };
    const auto beacon_potential = create_correlated_beacons(clique);

    return std::make_tuple(road_map, ekf_slam, beacon_potential);
}

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_circle_environment(
    const EkfSlamConfig &ekf_config, const int num_landmarks, const double circle_radius_m) {
    // Create an environment where the landmarks are arranged in a circle and the robot starts and
    // ends in the middle. The road map connects the landmarks to their neighbor and to the center.
    // The landmarks are numbered with the zeroth landmark at the +x axis and increasing in id
    // in the counter clockwise direction.
    // Create the Road Map
    const Eigen::Vector2d START_GOAL_STATE{0.0, 0.0};
    const int source_id = num_landmarks;
    const int sink_id = num_landmarks + 1;

    std::vector<Eigen::Vector2d> points;
    Eigen::MatrixXd adj(num_landmarks + 2, num_landmarks + 2);
    for (int i = 0; i < num_landmarks; i++) {
        const double x_pos_m =
            circle_radius_m * std::cos(2.0 * std::numbers::pi * i / num_landmarks);
        const double y_pos_m =
            circle_radius_m * std::sin(2.0 * std::numbers::pi * i / num_landmarks);
        points.push_back(Eigen::Vector2d{x_pos_m, y_pos_m});

        // Each landmark connects to the one preceding it and the source/sink ids
        const int neighbor_id = i == 0 ? (num_landmarks - 1) : i - 1;
        for (const int neighbor : {source_id, sink_id, neighbor_id}) {
            adj(i, neighbor) = 1;
            adj(neighbor, i) = 1;
        }
    }
    // Pushback for the source and sink states
    points.push_back(START_GOAL_STATE);
    points.push_back(START_GOAL_STATE);

    const auto road_map = planning::RoadMap(points, std::move(adj),
                                            {{.start = START_GOAL_STATE,
                                              .goal = START_GOAL_STATE,
                                              .connection_radius_m = circle_radius_m / 2.0}});

    // Create the EKF Slam
    std::vector<Eigen::Vector2d> beacon_in_local(points.begin(), points.end() - 1);
    std::vector<int> beacon_ids(num_landmarks);
    std::iota(beacon_ids.begin(), beacon_ids.end(), 0);

    EkfSlam ekf_slam(ekf_config, time::RobotTimestamp());
    const MappedLandmarks mapped_landmarks = {
        .beacon_ids = beacon_ids,
        .beacon_in_local = beacon_in_local,
        .cov_in_local = Eigen::MatrixXd::Identity(2 * num_landmarks, 2 * num_landmarks) * 1e-12};
    const bool LOAD_OFF_DIAGONALS = true;
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    // Create the Beacon Potential
    const auto beacon_potential = AnticorrelatedBeaconPotential{.members = beacon_ids};
    return std::make_tuple(road_map, ekf_slam, beacon_potential);
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////

MappedLandmarks create_david_landmarks() {
    constexpr int LONE_BEACON_ID = 1;
    const Eigen::Vector2d LONE_BEACON_IN_LOCAL{-12.5, 12.5};
    constexpr int START_STACKED_BEACON_ID = 2;
    const Eigen::Vector2d STACKED_BEACON_IN_LOCAL{-12.5, -12.5};
    constexpr int NUM_STACKED_BEACONS = 3;
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
//With multiple landmarks in different places
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_david_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon, const double p_no_stack_beacon,
    const double p_stacked_beacon) {
    // Create the environment depicted below
    //
    //
    //                            
    //                               
    //                                         +Y▲
    //                                           │
    //                                           └──►
    //                  Start  5m               +X
    //                    X─────────X─────────X─────────X─────────X
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                 5m │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    X─────────X─────────X─────────X─────────X
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │ (0,0)   |         |
    //                    X─────────X─────────X─────────X─────────X
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    X─────────X─────────X─────────X─────────X
    //                   5│        6│        7│        8|        9|
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    │         │         │         |         |
    //                    X─────────X─────────X─────────X─────────X
    //                   0         1         2         3         4 END
    // Note that:
    //  - each X is a PRM node,
    //  - B represents a beacon,
    //  - The world origin exists at the middle prm node,
    //  - The robot starts at (0, 10)
    //  - node indices start in the lower left and increase to the right, then increase up
        const auto mapped_landmarks = create_david_landmarks();
        const auto road_map = create_grid_road_map({10,10},{10,-10},5,5);
        auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
        constexpr bool LOAD_OFF_DIAGONALS = true;

        // Lone beacon potential
        const double lone_log_norm = -std::log(1 - p_lone_beacon);
        const double lone_param = std::log(p_lone_beacon) + lone_log_norm;
        const auto lone_potential =
        PrecisionMatrixPotential{.precision = Eigen::Matrix<double, 1, 1>{lone_param},
                                 .log_normalizer = lone_log_norm,
                                 .members = {1}};

        // Stacked Potential
        const auto stacked_potential = create_correlated_beacons({
            .p_beacon = p_stacked_beacon,
            .p_no_beacons = p_no_stack_beacon,
            .members = {3, 4, 5},
        });
        const auto beacon_potential = lone_potential * stacked_potential;

        // Move the robot to (-10, 10) and have it face to the right
        const liegroups::SE2 old_robot_from_new_robot(0, {-10, 10});
        ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
        ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

        return {road_map, ekf_slam, beacon_potential};
    }
/*
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_david_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon) {
        const auto mapped_landmarks = create_grid_mapped_landmarks({{12.5,12.5}},{1});
        const auto road_map = create_grid_road_map({10,-10},{-5,10},5,5);
        auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
        constexpr bool LOAD_OFF_DIAGONALS = true;

        // Lone beacon potential
        const double lone_log_norm = -std::log(1 - p_lone_beacon);
        const double lone_param = std::log(p_lone_beacon) + lone_log_norm;
        const auto lone_potential =
        PrecisionMatrixPotential{.precision = Eigen::Matrix<double, 1, 1>{lone_param},
                                 .log_normalizer = lone_log_norm,
                                 .members = {GRID_BEACON_ID}};
                     
        // Move the robot to (-10, 10) and have it face to the right
        const liegroups::SE2 old_robot_from_new_robot(0, {-10, 10});
        ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
        ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

        return {road_map, ekf_slam, lone_potential};
    }
*/
}  // namespace robot::experimental::beacon_sim

