#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include <unordered_map>

#include "experimental/beacon_sim/beacon_potential.hh"
#include "common/math/logsumexp.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/road_map.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"

namespace robot::planning {

struct DavidPlannerConfigThree {
    const int max_visits;
    const int max_plans;
    const double max_sensor_range_m;
    std::optional<time::RobotTimestamp::duration> timeout;
    std::optional<double> uncertainty_tolerance;
};

template <typename State>
struct SuccessorThree {
    State state; 
    double edge_cost;
    double weight;
};

template <typename State>
struct DavidPlannerResultThree {
    std::vector<State> nodes;
    double log_probability_mass_tracked;
};

template <typename State, typename SuccessorFunc>
std::optional<State> pick_successor_three(const State &curr_state, const SuccessorFunc &successor_for_state,
                    InOut<std::mt19937> gen, const double &favor_goal, std::unordered_map<State,int> *states,
                    const DavidPlannerConfigThree config, std::unordered_map<int,Eigen::Vector2d> *local_beacon_coords,
                    const std::vector<Eigen::Vector2d> &node_coords,
                    const RoadMap &road_map) {
    
    std::vector<SuccessorThree<State>> scsrs;
    Eigen::Vector2d scsr_coords;
    //std::cout << " At State: " << curr_state << std::endl;
    // Making sure there is a traversable path
    // Finding weight of each successor based on beacons in map
    // Only adding to potential scsr picks if the max_visits hasn't been reached
    bool check = false;
    double total_weight = 0;
    for(auto &scsr : successor_for_state(curr_state)){
        if(states->at(scsr.state) == config.max_visits){continue;}
        check = true;
        scsr_coords = node_coords[scsr.state];
        for(auto &bcn : *local_beacon_coords){scsr.weight += pow(sqrt(pow((bcn.second[0] - scsr_coords[0]),2)+pow((bcn.second[1]-scsr_coords[1]),2)),-2);} // Add beacon_potential??
        scsrs.push_back(scsr);
        total_weight+=scsr.weight;
    }
    if(check == false){return std::nullopt;}

    // Adding weight to the closest scsr(s) to the goal
    // Finding which successor is closest to goal
    double dist_to_goal = 1e4;
    double maybe_dist_to_goal = 1e5;
    State closest_scsr;
    //Eigen::Vector2d goal = node_coords[-2];
    //Eigen::Vector2d goal = road_map.point;
    Eigen::Vector2d goal = road_map.point(RoadMap::GOAL_IDX);
    for(auto &scsr : scsrs){
        scsr_coords = node_coords[scsr.state];
        if(scsr.state == -2){maybe_dist_to_goal = 0;
        }else{maybe_dist_to_goal = sqrt(pow((goal[0] - scsr_coords[0]),2)+pow((goal[1]-scsr_coords[1]),2));}
        if(maybe_dist_to_goal < dist_to_goal){dist_to_goal = maybe_dist_to_goal, closest_scsr = scsr.state;}
    }
    // Adding weight to closest scsr to goal based on favor_goal desire.
    //Calculating new total with favor goal weight added
    for(auto &scsr : scsrs){
        if(scsr.state == closest_scsr){
            double goal_weight = (total_weight/(1-favor_goal+1e-12)) - total_weight; // "1e-12" so that its never 1/0
            scsr.weight += goal_weight;
        }
    }
    total_weight = total_weight/(1-favor_goal+1e-12);

    // Normalizing scsrs
    std::tuple<State,double> p_scsr;
    std::vector<std::tuple<State,double>> p_scsrs;
    for(auto &scsr :scsrs){
        get<0>(p_scsr) = scsr.state;
        get<1>(p_scsr) = scsr.weight / total_weight;
        p_scsrs.push_back(p_scsr);
    }

    // Printing scsrs for testing purposes
    //std::cout << "[";
    //for(auto &scsr : p_scsrs){
    //    if(scsr == p_scsrs[-1]){std::cout << get<0>(scsr) << ":" << std::setprecision(5) << 100*get<1>(scsr);
    //    }else{std::cout << get<0>(scsr) << ":" << std::setprecision(5) << 100*get<1>(scsr) << "%,";}
    //}
    //std::cout << "]" << std::endl;

    // Picking a successor
    std::uniform_real_distribution<> dist;
    State picked_scsr = -2;
    double rand_num = dist(*gen);
    std::tuple<State,double> val;
    for (int k = 0; k < (int)p_scsrs.size(); k++) {
        val = p_scsrs.at(k);
        if(rand_num < get<1>(val)) {
            picked_scsr = get<0>(val);
            break;
        }
        rand_num -= get<1>(val);
    }

    // Remove beacon for beacon coords if it is within max_view_range_m of picked scsr
    std::vector<int> beacons_to_remove;
    Eigen::Vector2d picked_scsr_coords = node_coords[picked_scsr];
    double distance;
    for(auto &bcn : *local_beacon_coords){
        distance = sqrt(pow((bcn.second[0] - picked_scsr_coords[0]),2)+pow((bcn.second[1] - picked_scsr_coords[1]),2));
        if(distance < config.max_sensor_range_m){beacons_to_remove.push_back(bcn.first);}
    }
    states->at(picked_scsr)+=1;
    return picked_scsr;
}

template <typename State, typename SuccessorFunc>
std::optional<std::vector<State>> sample_path_three(
    const DavidPlannerConfigThree &config,
    const RoadMap &road_map, InOut<std::mt19937> gen,
    const double &favor_goal, const SuccessorFunc &successor_for_state,
    const std::unordered_map<int,Eigen::Vector2d> &beacon_coords,
    const std::vector<Eigen::Vector2d> &node_coords) {
    std::vector<State> plan;
    State curr_state = road_map.START_IDX;
    const State GOAL_STATE = road_map.GOAL_IDX;
    auto local_beacon_coords = beacon_coords;

    std::unordered_map<State,int> states;
    states[road_map.START_IDX] = config.max_visits;
    states[road_map.GOAL_IDX] = 0;
    for(int j=0; j<((int)road_map.points().size());j++){states[j] = 0;}

    plan.push_back(curr_state);
    while(curr_state != GOAL_STATE) {
        auto next_state = pick_successor_three<State>(curr_state,successor_for_state,gen,favor_goal,&states,config,&local_beacon_coords, node_coords, road_map);
        if(next_state.has_value()){
            plan.push_back(next_state.value());
            curr_state = next_state.value();
        }else{return std::nullopt;}
    }
    return plan;
}

template <typename State, typename SuccessorFunc>
std::optional<DavidPlannerResultThree<State>> david_planner_three( 
    const SuccessorFunc &successor_for_state,
    const RoadMap &road_map, const DavidPlannerConfigThree config,
    const experimental::beacon_sim::EkfSlamEstimate &ekf_estimate,
    const experimental::beacon_sim::BeaconPotential &beacon_potential,
    const experimental::beacon_sim::EkfSlam &ekf,const double &favor_goal) {

    const time::RobotTimestamp plan_start_time = time::current_robot_time();    
    std::mt19937 gen(44); //generate with respect to real time?
    std::vector<std::vector<State>> world_samples;
    std::vector<double> log_probs;
    // Sampling worlds    
    world_samples.reserve(config.max_plans);
    for(int i=0; i < (int)config.max_plans; i++){
        world_samples.emplace_back(beacon_potential.sample(make_in_out(gen)));
        const auto &sample = world_samples.back();
        log_probs.push_back(beacon_potential.log_prob(sample));

        // Create a potential with only this assignment
        std::unordered_map<int, bool> assignment;
        for (const int beacon_id : beacon_potential.members()) {
            const auto iter = std::find(sample.begin(), sample.end(), beacon_id);
            const bool is_beacon_present_in_sample = iter != sample.end();
            assignment[beacon_id] = is_beacon_present_in_sample;
        }
        const experimental::beacon_sim::BeaconPotential conditioned = beacon_potential.conditioned_on(assignment);
    }
    // Sampling plans
    std::vector<int> beacon_ids = ekf_estimate.beacon_ids;
    //std::vector<Eigen::Vector2d> beacon_coords;
    std::unordered_map<int,Eigen::Vector2d> beacon_coords;

    for (int i = 0; i < (int)beacon_ids.size(); i++) {
        const std::optional<Eigen::Vector2d> maybe_beacon_coords = ekf_estimate.beacon_in_local(beacon_ids[i]);
        if (maybe_beacon_coords.has_value()) {
            beacon_coords[beacon_ids[i]] = maybe_beacon_coords.value();
        }
    }
    std::vector<Eigen::Vector2d> node_coords = road_map.points();

    std::vector<std::vector<State>> plans;
    while ((int)plans.size() < config.max_plans) {
        auto plan = sample_path_three<State>(config,road_map,make_in_out(gen),favor_goal,successor_for_state,beacon_coords,node_coords);
        if(plan.has_value()) {
            plans.push_back(plan.value());
        }
    }
    // Evaluate plans on worlds
    std::vector<double> expected_cov_dets(plans.size(), 0.0);
    for (const auto &sample : world_samples) {
        const auto covs = experimental::beacon_sim::evaluate_paths_with_configuration(plans, ekf, road_map, config.max_sensor_range_m, sample);

        for (int i = 0; i < (int)plans.size(); i++) {
            expected_cov_dets.at(i) += covs.at(i).cov_in_robot.determinant() / world_samples.size();
        }
        if (config.timeout.has_value() &&
            time::current_robot_time() - plan_start_time > config.timeout.value()) {
            break;
        }
    }
    const auto min_iter = std::min_element(expected_cov_dets.begin(), expected_cov_dets.end());
    const int min_idx = std::distance(expected_cov_dets.begin(), min_iter);

    return {{
        .nodes = plans.at(min_idx),
        .log_probability_mass_tracked = math::logsumexp(log_probs),
    }};
    }
}  // namespace robot::planning
