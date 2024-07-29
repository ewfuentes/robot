#include "planning/david_planning_three.hh"
#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"


namespace robot::planning {
    TEST(DavidPlannerTest, testing_evaluation){

        // SETUP
        const experimental::beacon_sim::EkfSlamConfig ekf_config{
        .max_num_beacons = 3,
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

        const double P_FIRST_BEACON = 0.4;
        const double P_SECOND_BEACON = 0.4;
        const double P_THIRD_BEACON = 0.9;
        const auto &[grid, ekf_slam, potential] = create_david_grid_environment(ekf_config,P_FIRST_BEACON,P_SECOND_BEACON,P_THIRD_BEACON);
        const auto road_map = grid;
        const auto beacon_potential = potential;
        const auto ekf = ekf_slam;

        const experimental::beacon_sim::EkfSlamEstimate &est = ekf_slam.estimate();
        const DavidPlannerConfigThree david_config{
            .max_visits = 1,
            .max_plans = 1,
            .max_sensor_range_m = 4,
        };

        auto successor_func = [&road_map](const int &node_idx) -> std::vector<SuccessorThree<int>> {
            std::vector<SuccessorThree<int>> out;
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

        // ACTION
        const double FAVOR_GOAL = .5;
        const auto maybe_plan = david_planner_three<int>(successor_func, road_map,david_config,est,beacon_potential,ekf,FAVOR_GOAL);
        //VERIFICATION

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