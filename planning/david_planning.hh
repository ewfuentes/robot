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
#include "planning/djikstra.hh"
#include "planning/road_map.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"

namespace robot::planning {

struct DavidPlannerConfig {
    const int max_plans;
    const int max_states_per_path;
    const double favor_goal;
    const double favor_exploration;
    const double edge_cost;
    const int num_configuration_samples;
    int max_num_edge_transforms;
    std::optional<time::RobotTimestamp::duration> timeout;
    const double max_sensor_range_m;
    std::optional<double> uncertainty_tolerance;
};

template <typename State>
struct Beacon {
        State node;
        DjikstraResult<State> djikstra;
        bool visited;
};

template <typename State>
struct DavidPlannerResult {
    std::vector<State> nodes;
    double log_probability_mass_tracked;
};

template <typename State, typename SuccessorFunc, typename TerminationCheck>
std::unordered_map<State,Beacon<State>> beacon_mapping(
    const SuccessorFunc &successor_for_state, const TerminationCheck &termination_check,
    const RoadMap &road_map, const experimental::beacon_sim::EkfSlamEstimate ekf_estimate) {
    std::vector<State> beacon_ids = ekf_estimate.beacon_ids;
    std::vector<Eigen::Vector2d> beacon_coords;

    for (int i = 0; i < (int)beacon_ids.size(); i++) {
        const std::optional<Eigen::Vector2d> maybe_beacon_coords =
            ekf_estimate.beacon_in_local(beacon_ids[i]);
        if (maybe_beacon_coords.has_value()) {
            beacon_coords.push_back(maybe_beacon_coords.value());
        }
    }

    std::vector<Eigen::Vector2d> node_coords = road_map.points();
    std::unordered_map<State,Beacon<State>> beacon_map;
    Beacon<State> beacon_item;
    int j = 0;
    int pot_beacon_node = -200;
    for (const auto &beacon : beacon_coords) {
        double dist = 1e10;
        for (int i = 0; i < (int)node_coords.size(); i++) {
            const auto curr_node = node_coords[i];
            double pot_dist = sqrt(pow(curr_node[0] - beacon[0], 2) + pow(curr_node[1] - beacon[1], 2));
            if (pot_dist < dist) {
                dist = pot_dist;
                pot_beacon_node = i;
            }
        }
        beacon_item.node = pot_beacon_node;
        beacon_item.djikstra = djikstra<State>(beacon_item.node, successor_for_state, termination_check);
        beacon_item.visited = false;
        beacon_map[beacon_ids[j]] = beacon_item;
        j++;
    }
    return beacon_map;
}

template <typename State, typename SuccessorFunc>
std::tuple<State,std::unordered_map<State,Beacon<State>>> pick_successor(const State &curr_state, const SuccessorFunc &successor_for_state,
                     const DavidPlannerConfig &config, const DjikstraResult<State> &djikstra_goal, InOut<std::mt19937> gen, 
                     const experimental::beacon_sim::BeaconPotential &beacon_potential, std::unordered_map<State,Beacon<State>> &beacon_map) {
    std::tuple<State,std::unordered_map<State,Beacon<State>>> out;                    
    std::unordered_map<State,Beacon<State>> local_beacon_map = beacon_map;

    std::vector<Successor<State>> goal_successors;
    std::vector<Successor<State>> exploration_successors;
    std::vector<std::tuple<State, double>> p_all_successors; // currently there is no check to make sure sum of all probabilities is 1.0
    std::tuple<State, double> p_successor;

    // Finding the minimum cost to the goal from successors
    double min_goal_cost = 1e10;
    for(const auto &successor : successor_for_state(curr_state)){
        if(djikstra_goal.cost_to_go_from_state.at(successor.state) < min_goal_cost){
            min_goal_cost = djikstra_goal.cost_to_go_from_state.at(successor.state);
        }
    }
    // Assigning successors to either goal group or exploratory group
    for (const auto &successor : successor_for_state(curr_state)) {
        if(djikstra_goal.cost_to_go_from_state.at(successor.state) == min_goal_cost){
            goal_successors.push_back(successor);
        }else{
            exploration_successors.push_back(successor);
        }
    }

    // Goal successors normalization
    bool successor_check = false;
    for(const auto &successor : exploration_successors){
        if(successor.state == -1){
            successor_check = true;
        }
    }
    if(successor_check == true){
        for(const auto &successor : goal_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = 1.00/(double)goal_successors.size();// can potential be determined by distance to beacon
            p_all_successors.push_back(p_successor);
        }
    }else{
        for(const auto &successor : goal_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = (1.00/(double)goal_successors.size())*config.favor_goal;// can potential be determined by distance to beacon
            p_all_successors.push_back(p_successor);
        }
    }
    // up until this point the only improves I see coming are using iterators instead of loops and changing the noralization of the goal successors

    // Exploration successors normalization
    double total_cost = 0.0;
    // Gathering total cost of all successors
    std::unordered_map<int,bool> p_beacon_pass_in;
    for(const auto &successor : exploration_successors){
        double closest_beacon = 1e15;
        State closest_beacon_id;
        for(const auto &[key,beacon] : local_beacon_map){
            if(beacon.djikstra.cost_to_go_from_state.at(successor.state) < closest_beacon){
                closest_beacon = beacon.djikstra.cost_to_go_from_state.at(successor.state);
                closest_beacon_id = key;
            }
        }
        p_beacon_pass_in.clear();
        p_beacon_pass_in[closest_beacon_id] = true; // have to redclare everytime so it doesnt pass in the combined probabilities 

        if((closest_beacon == 0) && (local_beacon_map.at(closest_beacon_id).visited == false)){
            total_cost += 1/(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)));
        }else if((closest_beacon == 0) && (local_beacon_map.at(closest_beacon_id).visited == true)){ //could change to remove successor from exploratory list next
            total_cost += 1/(config.edge_cost*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true))));
        }else{
            total_cost += 1/(closest_beacon*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true))));
        }
    }


    // Assigning probabilities to each exploratory successor
    std::vector<std::tuple<State, double>> p_explore_successors;
    double closest_beacon = 1e15; 
    State closest_beacon_id;
    for(const auto &successor : exploration_successors){
        closest_beacon = 1e15;
        for(const auto &[key,beacon] : local_beacon_map){
            if(beacon.djikstra.cost_to_go_from_state.at(successor.state) < closest_beacon){
                closest_beacon = beacon.djikstra.cost_to_go_from_state.at(successor.state);
                closest_beacon_id = key;
            }
        }
        p_beacon_pass_in.clear();
        p_beacon_pass_in[closest_beacon_id] = true; 

        get<0>(p_successor) = successor.state;
        if((closest_beacon == 0) && (local_beacon_map.at(closest_beacon_id).visited == false)){
            get<1>(p_successor) = config.favor_exploration/(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true))*total_cost);
            local_beacon_map[closest_beacon_id].visited = true;
            p_explore_successors.push_back(p_successor);
        }else if((closest_beacon == 0) && (local_beacon_map.at(closest_beacon_id).visited == true)){ //could change to remove successor from exploratory list next
            get<1>(p_successor) = config.favor_exploration/(config.edge_cost*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost);
            p_explore_successors.push_back(p_successor);
        }else{
            get<1>(p_successor) = config.favor_exploration/(closest_beacon*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost);
            p_explore_successors.push_back(p_successor);
    }}
    if(successor_check == false){
        for(const auto &successor : p_explore_successors){
            p_all_successors.push_back(successor);
        }
    }else{
        for(auto &successor : p_explore_successors){
            get<1>(successor) = 0.0;
            p_all_successors.push_back(successor);
        }
    }

    std::cout << "At State: "  << curr_state<< std::endl;
    std::cout << "[";
    for(auto &successor : p_all_successors){
        std::cout << get<0>(successor) << ":" << std::setprecision(3) << 100*get<1>(successor) << "%,";
    }
    std::cout << "]";
    // Picking a successor
    std::uniform_real_distribution<> dist;

    double rand_num = dist(*gen);
    //std::cout << rand_num << std::endl;
    for (int k = 0; k < (int)p_all_successors.size(); k++) {
        const auto val = p_all_successors.at(k);
        if (rand_num < get<1>(val)) {
            get<0>(out) = get<0>(val);
            get<1>(out) = local_beacon_map;
            break;
        }
        rand_num -= get<1>(val);
    }
    return out;
}

template <typename State, typename SuccessorFunc>
std::optional<std::vector<State>> sample_path(
    const DjikstraResult<State> &djikstra_goal,
    std::unordered_map<State,Beacon<State>> &beacon_map,
    const SuccessorFunc &successor_for_state, const DavidPlannerConfig &config,
    const RoadMap &road_map,InOut<std::mt19937> gen,
    const experimental::beacon_sim::BeaconPotential &beacon_potential) {
    std::vector<State> path;
    State curr_state = road_map.START_IDX;
    const State GOAL_STATE = road_map.GOAL_IDX;

    while(curr_state != GOAL_STATE) {
        if ((int)path.size() > config.max_states_per_path) {
            return std::nullopt;
        }
        const auto next_state = pick_successor<int>(curr_state,successor_for_state,config,djikstra_goal,gen,beacon_potential,beacon_map);
        beacon_map = get<1>(next_state);
        int picked_successor = get<0>(next_state);
        path.push_back(picked_successor);
        curr_state = picked_successor;
    }
    return path;
}

template <typename State, typename SuccessorFunc, typename TerminationCheck>
std::optional<DavidPlannerResult<State>> david_planner( 
    const SuccessorFunc &successor_for_state, const TerminationCheck &termination_check,
    const RoadMap &road_map, const DavidPlannerConfig &config,
    const experimental::beacon_sim::EkfSlamEstimate &ekf_estimate,
    const experimental::beacon_sim::BeaconPotential &beacon_potential,
    const experimental::beacon_sim::EkfSlam &ekf) {

    const time::RobotTimestamp plan_start_time = time::current_robot_time();    
    std::mt19937 gen(111211); //generate with respect to real time?
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
    std::vector<std::vector<State>> plans;
    const auto djikstra_goal = djikstra<State>(-2, successor_for_state, termination_check);
    auto beacon_map = beacon_mapping<State>(successor_for_state, termination_check, road_map, ekf_estimate);
    while ((int)plans.size() < config.max_plans) {
        const auto plan = sample_path<State>(djikstra_goal, beacon_map, successor_for_state, config, road_map,make_in_out(gen),beacon_potential);
        if (plan.has_value()) {
            plans.push_back(plan.value());
        }

    }
    // Evaluate plans on worlds
    std::vector<double> expected_cov_dets(plans.size(), 0.0);
    for (const auto &sample : world_samples) {
        const auto covs = evaluate_paths_with_configuration(plans, ekf, road_map, config.max_sensor_range_m, sample);

        for (int i = 0; i < (int)plans.size(); i++) {
            expected_cov_dets.at(i) += covs.at(i).determinant() / world_samples.size();
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
