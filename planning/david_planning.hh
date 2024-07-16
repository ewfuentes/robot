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
    const int max_visits;
    const int max_plans;
    int max_num_edge_transforms;
    std::optional<time::RobotTimestamp::duration> timeout;
    const double max_sensor_range_m;
    std::optional<double> uncertainty_tolerance;
};

template <typename State>
struct Beacon {
        State node;
        double dist_to_node;
        DjikstraResult<State> djikstra;
};

template <typename State>
struct DavidPlannerResult {
    std::vector<State> nodes;
    double log_probability_mass_tracked;
};



template <typename State, typename SuccessorFunc, typename TerminationCheck>
std::unordered_map<State,Beacon<State>> beacon_mapping(
    const SuccessorFunc &successor_for_state, const TerminationCheck &termination_check,
    const RoadMap &road_map, const experimental::beacon_sim::EkfSlamEstimate ekf_estimate,
    const DavidPlannerConfig &config) {
    std::vector<int> beacon_ids = ekf_estimate.beacon_ids;
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
    State pot_beacon_node;
    for (const auto &beacon : beacon_coords) {
        double dist = 1e10;
        for (int i = 0; i < (int)node_coords.size(); i++) {
            const auto curr_node = node_coords[i];
            double pot_dist = sqrt(pow(curr_node[0] - beacon[0], 2) + pow(curr_node[1] - beacon[1], 2));
            if (pot_dist < dist) {dist = pot_dist, pot_beacon_node = i;}
        }
        if(dist <= config.max_sensor_range_m){
            beacon_item.node = pot_beacon_node;
            beacon_item.dist_to_node = dist;
            beacon_item.djikstra = djikstra<State>(beacon_item.node, successor_for_state, termination_check);
            beacon_map[beacon_ids[j]] = beacon_item;
        }
        j++;
    }
    return beacon_map;
}

template <typename State>
std::vector<std::tuple<State, double>> sort_explore_probabilities(
                    std::unordered_map<State,Beacon<State>> *beacon_map, 
                    std::vector<Successor<State>> exploration_successors,
                    std::vector<Successor<State>> goal_successors,
                    const double &favor_goal,
                    const experimental::beacon_sim::BeaconPotential &beacon_potential){

    std::vector<std::tuple<State, double>> p_explore_successors;
    std::tuple<State, double> p_successor;

    if(beacon_map->empty()){
        for(auto &successor : exploration_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = (1.00/(double)exploration_successors.size())*(1-favor_goal);
            p_explore_successors.push_back(p_successor);
        }
        return p_explore_successors;
    }else{
        //Finds the bottom-line closest beacon to every successor
        for(auto &successor : exploration_successors){
            double closest_beacon = 1e4;
            for(const auto &[key,beacon] : *beacon_map){
                if((beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node) < closest_beacon){
                    closest_beacon = beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node;
                    successor.nearest_beacon_id = key;
                }
            }
            successor.nearest_beacon_dist = closest_beacon;
        }
        for(auto &successor : exploration_successors){
            std::cout << "Successor: " << successor.state << ", " << successor.nearest_beacon_id.value() << ", " << successor.nearest_beacon_dist << std::endl;
        }
        // Makes sure multiple successors aren't mapped to same beacon unless they are equidistant from the beacon and no other equidistance beacon is present
        
        bool successors_unique = false; //If still false by end of while loop, reloops
        while(!successors_unique){
            successors_unique = true;
            for(auto &successor : exploration_successors){
                for(auto &successor_neighbor : exploration_successors){
                    if(successor.state == successor_neighbor.state || !successor.nearest_beacon_id.has_value() || !successor_neighbor.nearest_beacon_id.has_value()){continue;
                    }else if(successor.nearest_beacon_id.value() == successor_neighbor.nearest_beacon_id.value()){
                        if(successor.nearest_beacon_dist > successor_neighbor.nearest_beacon_dist){
                            if((int)beacon_map->size() == 1){
                                successor.nearest_beacon_id = std::nullopt; //change to std::optional and pass nullopt if care, cant find log prob of made up id
                                successor.nearest_beacon_dist = 1e10; //figure out how to assign probability to some aggregiously high number
                                break;
                            }else{
                                double closest_beacon = 1e10;
                                int curr_beacon_id = successor.nearest_beacon_id.value();
                                for(const auto &[key,beacon] : *beacon_map){
                                    if(key == curr_beacon_id){continue;
                                    }else if(beacon_map->size() == 1){
                                        successor.nearest_beacon_dist = 1e3; 
                                        break;
                                    }else if((beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node) < closest_beacon){
                                        closest_beacon = beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node;
                                        successor.nearest_beacon_id = key;
                                    }
                                }
                                successor.nearest_beacon_dist = closest_beacon;
                                std::cout << "ran " << std::endl;
                            }
                        }else if(successor.nearest_beacon_dist == successor_neighbor.nearest_beacon_dist){continue;
                        }else{successors_unique = false;}
                    }
                }
            }
        }
        double total_cost = 0.0;
        std::unordered_map<State,bool> p_beacon_pass_in; // A necessary condition for beacon_potential.log_prob(...)
        for(auto &successor : exploration_successors){
            p_beacon_pass_in.clear();
            if(successor.nearest_beacon_id.has_value()){
                p_beacon_pass_in[successor.nearest_beacon_id.value()] = true;
                total_cost += 1/((successor.nearest_beacon_dist+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true))));
            }else{
                total_cost += 1/(successor.nearest_beacon_dist+1e-2);
            }
        }
        for(auto &successor : exploration_successors){
            p_beacon_pass_in.clear();
            get<0>(p_successor) = successor.state;
            if((int)goal_successors.size()==0){
                p_beacon_pass_in[successor.nearest_beacon_id.value()] = true;
                get<1>(p_successor) = 1/((successor.nearest_beacon_dist+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost);
            }else{
                if(successor.nearest_beacon_id.has_value()){
                    p_beacon_pass_in[successor.nearest_beacon_id.value()] = true; 
                    get<1>(p_successor) = (1-favor_goal)/((successor.nearest_beacon_dist+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost);
                }else{get<1>(p_successor) = (1-favor_goal)/((successor.nearest_beacon_dist+1e-2)*total_cost);}
                p_explore_successors.push_back(p_successor);
            }
        }
    }
    return p_explore_successors;
   /*
    std::vector<std::tuple<State, double>> p_explore_successors;
    // Keep this
    auto local_beacon_map = beacon_map;
    std::tuple<State, double> p_successor;

    double total_cost = 0.0;
    if(local_beacon_map->empty()){
        for(const auto &successor : exploration_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = (1.00/(double)exploration_successors.size())*(1-favor_goal);
            p_explore_successors.push_back(p_successor);
        }
    }else{ // this is where add beacon mapping to successors
        std::unordered_map<State,bool> p_beacon_pass_in;
        for(const auto &successor : exploration_successors){
            double closest_beacon = 1e4;
            State closest_beacon_id;
            for(const auto &[key,beacon] : *local_beacon_map){
                if((beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node) < closest_beacon){
                    closest_beacon = beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node;
                    closest_beacon_id = key;
                }
            }
            //Keep for now
            // have to clear and redeclare everytime so it doesnt pass in the combined probabilities
            p_beacon_pass_in.clear();
            p_beacon_pass_in[closest_beacon_id] = true;
            //Adding 1e-2 to closest_beacon because value could be zero which would make total cost infinite
            total_cost += 1/((closest_beacon+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true))));
        }
        // Keep for now
        // Assigning probabilities to each exploratory successor
        double closest_beacon;
        State closest_beacon_id;
        for(const auto &successor : exploration_successors){
            closest_beacon = 1e15;
            for(const auto &[key,beacon] : *local_beacon_map){
                if((beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node) < closest_beacon){
                    closest_beacon = (beacon.djikstra.cost_to_go_from_state.at(successor.state)+beacon.dist_to_node);
                    closest_beacon_id = key;
                }
            }
            p_beacon_pass_in.clear();
            p_beacon_pass_in[closest_beacon_id] = true; 
            get<0>(p_successor) = successor.state;

            ((int)goal_successors.size()==0) ? 
            get<1>(p_successor) = 1/((closest_beacon+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost) :
            get<1>(p_successor) = (1-favor_goal)/((closest_beacon+1e-2)*(1-exp(beacon_potential.log_prob(p_beacon_pass_in,true)))*total_cost);
            p_explore_successors.push_back(p_successor);
        }
    }
    return p_explore_successors;
    */
}


template <typename State, typename SuccessorFunc>
std::optional<State> pick_successor(const State &curr_state, const SuccessorFunc &successor_for_state,
                    const DjikstraResult<State> &djikstra_goal, InOut<std::mt19937> gen, 
                    const experimental::beacon_sim::BeaconPotential &beacon_potential, 
                    std::unordered_map<State,Beacon<State>> *beacon_map,
                    const double &favor_goal, std::unordered_map<State,int> *states,
                    const DavidPlannerConfig config) {
    State out;                    
    std::vector<Successor<State>> goal_successors;
    std::vector<Successor<State>> exploration_successors;
    std::vector<std::tuple<State, double>> p_all_successors;
    std::tuple<State, double> p_successor;

    // Finding the minimum cost to the goal from successors
    double min_goal_cost = 1e10; // Adjust this to include multiple beacon distances
    for(const auto &successor : successor_for_state(curr_state)){
        if(djikstra_goal.cost_to_go_from_state.at(successor.state) < min_goal_cost){
            min_goal_cost = djikstra_goal.cost_to_go_from_state.at(successor.state);
        }
    }
    // Assigning successors to either goal group or exploratory group
    for (const auto &successor : successor_for_state(curr_state)) {
        if(states->at(successor.state) == config.max_visits){continue;}
        if(djikstra_goal.cost_to_go_from_state.at(successor.state) == min_goal_cost){goal_successors.push_back(successor);
        }else{exploration_successors.push_back(successor);}
    }
    if(((int)exploration_successors.size()==0) && ((int)goal_successors.size()==0)){return std::nullopt;}

    // Goal successors normalization
    if(exploration_successors.size()==0){
        for(const auto &successor : goal_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = 1.00/(double)goal_successors.size();// can potential be determined by distance to beacon
            p_all_successors.push_back(p_successor);
        }
    }else{
        for(const auto &successor : goal_successors){
            get<0>(p_successor) = successor.state;
            get<1>(p_successor) = (1.00/(double)goal_successors.size())*favor_goal;// can potential be determined by distance to beacon
            p_all_successors.push_back(p_successor);
        }
    }
    // Exploration successors normalization
    const auto p_explore_successors = sort_explore_probabilities<State>(beacon_map, exploration_successors, goal_successors,favor_goal,beacon_potential);
    p_all_successors.insert(std::end(p_all_successors), std::begin(p_explore_successors), std::end(p_explore_successors));

    std::cout << "[";
    for(auto &successor : p_all_successors){
        std::cout << get<0>(successor) << ":" << std::setprecision(5) << 100*get<1>(successor) << "%,";
    }
    std::cout << "]" << std::endl;
    // Picking a successor
    std::uniform_real_distribution<> dist;
    State picked_node;
    double rand_num = dist(*gen);
    std::tuple<State,double> val;
    for (int k = 0; k < (int)p_all_successors.size(); k++) {
        val = p_all_successors.at(k);
        if(rand_num < get<1>(val)) {
            out = get<0>(val);
            picked_node = out;
            break;
        }
        rand_num -= get<1>(val);
    }
    std::vector<int> keys_to_remove;
    for (const auto &[key,beacon] : *beacon_map){
        if(picked_node == beacon.node){keys_to_remove.push_back(key);}
    }
    for(int i=0;i<(int)keys_to_remove.size();i++){
        beacon_map->erase(keys_to_remove[i]);}
    states->at(out)+=1;
    std::cout << "At State: "  << picked_node << std::endl;
    return out;
}

template <typename State, typename SuccessorFunc>
std::optional<std::vector<State>> sample_path(
    const DjikstraResult<State> &djikstra_goal,
    std::unordered_map<State,Beacon<State>> &beacon_map,
    const SuccessorFunc &successor_for_state, const DavidPlannerConfig &config,
    const RoadMap &road_map,InOut<std::mt19937> gen,
    const experimental::beacon_sim::BeaconPotential &beacon_potential, const double &favor_goal) {
    std::vector<State> path;
    auto beacon_map_local = beacon_map;
    State curr_state = road_map.START_IDX;
    const State GOAL_STATE = road_map.GOAL_IDX;

    std::unordered_map<State,int> states;
    states[road_map.START_IDX] = config.max_visits;
    states[road_map.GOAL_IDX] = 0;
    for(int j=0; j<((int)road_map.points().size());j++){states[j] = 0;}

    path.push_back(curr_state);
    while(curr_state != GOAL_STATE) {
        const auto next_state = pick_successor<State>(curr_state,successor_for_state,djikstra_goal,gen,beacon_potential,&beacon_map_local,favor_goal,&states,config);
        if(next_state.has_value()){
            path.push_back(next_state.value());
            curr_state = next_state.value();
        }else{return std::nullopt;}
    }
    return path;
}

template <typename State, typename SuccessorFunc, typename TerminationCheck>
std::optional<DavidPlannerResult<State>> david_planner( 
    const SuccessorFunc &successor_for_state, const TerminationCheck &termination_check,
    const RoadMap &road_map, const DavidPlannerConfig config,
    const experimental::beacon_sim::EkfSlamEstimate &ekf_estimate,
    const experimental::beacon_sim::BeaconPotential &beacon_potential,
    const experimental::beacon_sim::EkfSlam &ekf,const double &favor_goal) {

    const time::RobotTimestamp plan_start_time = time::current_robot_time();    
    std::mt19937 gen(2); //generate with respect to real time?
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
    auto beacon_map = beacon_mapping<State>(successor_for_state, termination_check, road_map, ekf_estimate, config);
    while ((int)plans.size() < config.max_plans) {
        const auto plan = sample_path<State>(djikstra_goal, beacon_map, successor_for_state, config, road_map,make_in_out(gen),beacon_potential,favor_goal);
        if (plan.has_value()) {
            plans.push_back(plan.value());
        }
    }
    // Evaluate plans on worlds
    std::vector<double> expected_cov_dets(plans.size(), 0.0);
    for (const auto &sample : world_samples) {
        const auto covs = experimental::beacon_sim::evaluate_paths_with_configuration(plans, ekf, road_map, config.max_sensor_range_m, sample);

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
