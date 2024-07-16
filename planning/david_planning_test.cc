#include "planning/david_planning.hh"
#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"


namespace robot::planning {
/*
TEST(DavidPlannerTest, david_planning_test) {
    // Setup
    const DavidPlannerConfig david_config{
        .max_paths = 100,
        .max_beacons = 2,
        .favor_goal = 0.8,
        .favor_exploration = 0.2,
        .edge_cost = 5.0;
    };
    std::mt19937 gen(0);
    const planning::RoadMap grid = create_grid_road_map({-15,-10},{15,10},5,5);
    auto successor_func = [&grid](const int &node_idx) -> std::vector<Successor<int>> {
        std::vector<Successor<int>> out;
        const std::vector<std::tuple<int, Eigen::Vector2d>> &vector = grid.neighbors(node_idx);

        for(auto tuple : vector){
            const Eigen::Vector2d a_coords = grid.point(node_idx);
            const Eigen::Vector2d b_coords = get<1>(tuple);
            double d = sqrt(pow((b_coords[0]-a_coords[0]),2)+pow((b_coords[1]-a_coords[1]),2));

            out.push_back({
                    .state = get<0>(tuple),
                    .edge_cost = d
                    });
        };
        return out;
    };
    auto termination_check = [](std::unordered_map<int, double>) { return false; };
    // Action
    const auto maybe_plan = david_planner(successor_func,termination_check,grid,david_config,make_in_out(gen));

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    std::cout << "Num Nodes: " << plan.nodes.size() << std::endl;
    for (int i = 0; i < static_cast<int>(plan.nodes.size()); i++) {
        std::cout << i << " idx: " << plan.nodes.at(i) << std::endl;
    }
}
*/
/*
    TEST(DavidPlannerTest, testing_plans){
        // SETUP
        const experimental::beacon_sim::EkfSlamConfig ekf_config{
        .max_num_beacons = 2,
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
        .on_map_load_heading_uncertainty_rad = 0.1,
        };
        const double P_FIRST_BEACON = 0.5;
        const double P_SECOND_BEACON = 0.5;
        const auto &[grid, ekf_slam, potential] = create_david_grid_environment(ekf_config,P_FIRST_BEACON,P_SECOND_BEACON);
        const auto road_map = grid;
        const auto beacon_potential = potential;
        const experimental::beacon_sim::EkfSlamEstimate &est = ekf_slam.estimate();
        const DavidPlannerConfig david_config{
            .max_plans = 1,
            .max_num_edge_transforms = std::numeric_limits<int>::max(),
            .timeout = std::nullopt,
            .max_sensor_range_m = 4,
            .uncertainty_tolerance = std::nullopt,
        };

        auto successor_func = [&road_map](const int &node_idx) -> std::vector<Successor<int>> {
            std::vector<Successor<int>> out;
            const std::vector<std::tuple<int, Eigen::Vector2d>> &vector = road_map.neighbors(node_idx);

            for(auto tuple : vector){
                const Eigen::Vector2d a_coords = road_map.point(node_idx);
                const Eigen::Vector2d b_coords = get<1>(tuple);
                double d = sqrt(pow((b_coords[0]-a_coords[0]),2)+pow((b_coords[1]-a_coords[1]),2));

                out.push_back({
                        .state = get<0>(tuple),
                        .edge_cost = d,
                        });
            };
            return out;
        };
        auto termination_check = [](std::unordered_map<int, double>) { return false; };

        // ACTION

        const auto beacons = beacon_mapping<int>(successor_func,termination_check,road_map,est,david_config);
        // VERIFICATION
        std::cout << "Elements in beacons: " << beacons.size() << std::endl;
        std::vector<int> beacon_ids = est.beacon_ids;
        std::unordered_map<int,bool> p_beacon_pass_in;
        for(int i=0;i<(int)beacon_ids.size();i++){
            std::cout << "Beacon #" << i+1 << ":" << std::endl;
            std::cout << "Beacon ID: " << beacon_ids[i] << std::endl;
            p_beacon_pass_in.clear();
            p_beacon_pass_in[beacon_ids[i]] = true;
            std::cout << "Beacon Probability: " << exp(beacon_potential.log_prob(p_beacon_pass_in,true)) << std:: endl;
            std::cout << "Beacon Mapped to Node: " << beacons.at(beacon_ids[i]).node << std::endl;
            std::cout << std::endl;

        }

    }
    */
    TEST(DavidPlannerTest, testing_evaluation){

        // SETUP
        const experimental::beacon_sim::EkfSlamConfig ekf_config{
        .max_num_beacons = 2,
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
        .on_map_load_heading_uncertainty_rad = 0.1,
        };

        const double P_FIRST_BEACON = 0.9;
        const double P_SECOND_BEACON = 0.01;
        const auto &[grid, ekf_slam, potential] = create_david_grid_environment(ekf_config,P_FIRST_BEACON,P_SECOND_BEACON);
        const auto road_map = grid;
        const auto beacon_potential = potential;
        const auto ekf = ekf_slam;

        const experimental::beacon_sim::EkfSlamEstimate &est = ekf_slam.estimate();
        const DavidPlannerConfig david_config{
            .max_visits = 2,
            .max_plans = 1,
            .max_sensor_range_m = 4,
        };

        auto successor_func = [&road_map](const int &node_idx) -> std::vector<Successor<int>> {
            std::vector<Successor<int>> out;
            const std::vector<std::tuple<int, Eigen::Vector2d>> &vector = road_map.neighbors(node_idx);

            for(auto tuple : vector){
                const Eigen::Vector2d a_coords = road_map.point(node_idx);
                const Eigen::Vector2d b_coords = get<1>(tuple);
                double d = sqrt(pow((b_coords[0]-a_coords[0]),2)+pow((b_coords[1]-a_coords[1]),2));

                out.push_back({
                        .state = get<0>(tuple),
                        .edge_cost = d,
                        });
            };
            return out;
        };
        auto termination_check = [](std::unordered_map<int, double>) { return false; };

        // ACTION
        const double FAVOR_GOAL = 0.7;
        const auto maybe_plan = david_planner<int>(successor_func,termination_check,road_map,david_config,est,beacon_potential,ekf,FAVOR_GOAL);
        //VERIFICATION

        // Printing beacons
        std::vector<int> beacon_ids = est.beacon_ids;
        std::unordered_map<int,bool> p_beacon_pass_in;
        const auto beacons = beacon_mapping<int>(successor_func,termination_check,road_map,est,david_config);
        for(int i=0;i<(int)beacon_ids.size();i++){
            std::cout << "Beacon #" << i+1 << ":" << std::endl;
            std::cout << "Beacon ID: " << beacon_ids[i] << std::endl;
            p_beacon_pass_in.clear();
            p_beacon_pass_in[beacon_ids[i]] = true;
            std::cout << "Beacon Probability: " << exp(beacon_potential.log_prob(p_beacon_pass_in,true)) << std:: endl;
            std::cout << "Beacon Mapped to Node: " << beacons.at(beacon_ids[i]).node << std::endl;
            std::cout << std::endl;

        }
        // Printing config
        std::cout << FAVOR_GOAL*100 << "% Goal oriented and " << (1-FAVOR_GOAL)*100 << "% exploratory" << std::endl;
        std::cout << "Sampled From " << david_config.max_plans << " Plans" << std::endl;
        std::cout << std::endl;
        // Printing optimal path
        std::cout << "Optimal Path: [";
        for(const auto &state : maybe_plan->nodes){
            if(state == maybe_plan->nodes.back()){
                std::cout << state << "]" << std::endl;
            }else{
                std::cout << state << ",";
            }
        }
        std::cout << "Log(p) mass tracked: " << maybe_plan->log_probability_mass_tracked << std::endl;
    }
}