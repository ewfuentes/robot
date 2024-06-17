#include "experimental/beacon_sim/david_environment.hh"
#include <numbers>
#include "experimental/beacon_sim/anticorrelated_beacon_potential.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/precision_matrix_potential.hh"
#include "planning/road_map.hh"


namespace robot::experimental::beacon_sim {

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

    std::tuple<planning::RoadMap, EkfSlam> create_david_grid_environment(
    const EkfSlamConfig &ekf_config){
        const auto road_map = create_grid_road_map({25,0},{-20,25},9,9);
        auto ekf_slam = EkfSlam(ekf_config, time::RobotTimestamp());
        constexpr bool LOAD_OFF_DIAGONALS = true;

        // Move the robot to (-10, 10) and have it face to the right
        const liegroups::SE2 old_robot_from_new_robot(std::numbers::pi/2, {-20, 25});
        ekf_slam.predict(time::RobotTimestamp(), old_robot_from_new_robot);
        ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

        return {road_map, ekf_slam};
    }

    std::tuple<planning::RoadMap, EkfSlam> create_david_diamond_environment(
    const EkfSlamConfig &ekf_config) {
    //RoadMap
    //
    //
    //               5(7.5,7.5)
    //               /\
    //              /  \
    //             /    \
    //            /      \
    //           3(5,2.5)_4(10,2.5)
    //          ╱│\      /|\
    //         ╱ │ \    / | \
    //        ╱  │  \  /  |  \
    //       ╱   │   \/   |   \
    //(0,0)S \   |   /\   |   / E(15,0)
    //        ╲  │  /  \  |  /
    //         ╲ │ /    \ | /
    //          ╲│/______\|/
    //           1(5,-2.5)2(10,-2.5)
    //            \      /
    //             \    /    
    //              \  /   
    //               \/   
    //               0(7.5,-7.5)
    //                    
    //
    //Length is a bit skewed in depiction due to m need for it to look good

    const planning::RoadMap road_map(
        std::vector<Eigen::Vector2d>{{7.5,-7.5}, {5,-2.5}, {10, -2.5},{5,2.5},{10,2.5},{7.5,7.5}},
        (Eigen::Matrix3d() << 0,1,1,0,0,0,
                              1,0,1,1,1,0,
                              1,1,0,1,1,0,
                              0,1,1,0,1,1,
                              0,1,1,1,0,1,
                              0,0,0,1,1,0).finished(),
        planning::StartGoalPair{.start = {0, 0}, .goal = {15, 0}, .connection_radius_m = 7.2});

    // Create EkfSlam - I assume this is where I construct planner
    const std::vector<int> BEACON_IDS{123, 456};
    EkfSlam ekf_slam(ekf_config, time::RobotTimestamp());
    const MappedLandmarks mapped_landmarks = {
        .beacon_ids = BEACON_IDS,
        .beacon_in_local = {{3, -4}, {3, 4}},
        .cov_in_local = Eigen::Matrix4d::Identity() * 1e-12,
    };
    const bool LOAD_OFF_DIAGONALS = true;
    ekf_slam.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);
    const liegroups::SE2 world_from_robot(0, {15, 0});
    ekf_slam.estimate().local_from_robot(world_from_robot);

    return std::make_tuple(road_map, ekf_slam);
}

}  // namespace robot::experimental::beacon_sim