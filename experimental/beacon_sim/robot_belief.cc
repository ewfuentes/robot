
#include "experimental/beacon_sim/robot_belief.hh"

#include <iostream>

namespace robot::experimental::beacon_sim {

std::ostream &operator<<(std::ostream &out, const RobotBelief &belief) {
    out << "cov det: " << belief.cov_in_robot.determinant() << std::endl;
    out << belief.cov_in_robot;
    return out;
}

std::ostream &operator<<(std::ostream &out, const LandmarkRobotBelief &belief) {
    double expected_cov = 0.0;
    for (const auto &[key, component] : belief.belief_from_config) {
        const double det = component.cov_in_robot.determinant();
        out << key << " prob: " << std::exp(component.log_config_prob) << " cov det: " << det
            << std::endl;
        out << component.cov_in_robot << std::endl;
        expected_cov += std::exp(component.log_config_prob) * det;
    }
    out << "Expected det: " << expected_cov;
    return out;
}
}  // namespace robot::experimental::beacon_sim
