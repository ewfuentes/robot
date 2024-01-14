
#pragma once

#include "Eigen/Core"
#include "common/liegroups/se2.hh"

namespace robot::experimental::beacon_sim {

struct RobotBelief {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix3d cov_in_robot;
};

struct LandmarkRobotBelief {
    struct LandmarkConditionedRobotBelief {
        Eigen::Matrix3d cov_in_robot;
        double log_config_prob;
    };
    liegroups::SE2 local_from_robot;
    // If the belief represents a subset of the total landmark configurations,
    // this field represents the log probability of tracked beliefs. The log probability
    // of configuration `c` is then log_probability_mass_tracked +
    // belief_from_config[c].log_config_prob
    double log_probability_mass_tracked;
    std::unordered_map<std::string, LandmarkConditionedRobotBelief> belief_from_config;
};

}  // namespace robot::experimental::beacon_sim
