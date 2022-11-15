
#include "experimental/beacon_sim/belief_road_map_planner.hh"

namespace robot::experimental::beacon_sim {
namespace {
planning::BeliefUpdater<RobotBelief> make_belief_updater() {
    return [](const RobotBelief &initial_belief, const int start_idx, const int end_idx) {
        (void)start_idx;
        (void)end_idx;
        return initial_belief;
    };
}

}  // namespace

double distance_to(const Eigen::Vector2d &pt_in_local, const RobotBelief &belief) {
    const Eigen::Vector2d pt_in_robot = belief.local_from_robot.inverse() * pt_in_local;
    return pt_in_robot.norm();
}

double uncertainty_size(const RobotBelief &belief) {
    // Should this be the covariance about the map frame
    return belief.cov_in_robot.determinant();
}

bool operator==(const RobotBelief &a, const RobotBelief &b) {
    constexpr double TOL = 1e-6;
    const auto mean_diff = (a.local_from_robot.log() - b.local_from_robot.log()).norm();
    const auto cov_diff = (a.cov_in_robot - b.cov_in_robot).norm();

    const bool is_mean_near = mean_diff < TOL;
    const bool is_cov_near = cov_diff < (TOL * TOL);
    return is_mean_near && is_cov_near;
}

planning::BRMPlan<RobotBelief> compute_belief_road_map_plan(const planning::RoadMap &road_map,
                                                            const EkfSlam &ekf,
                                                            const Eigen::Vector2d &goal_state) {
    const auto &estimate = ekf.estimate();

    const RobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .cov_in_robot = estimate.robot_cov(),
    };

    const auto belief_updater = make_belief_updater();
    planning::plan<RobotBelief>(road_map, initial_belief, belief_updater, goal_state);

    return {};
}
}  // namespace robot::experimental::beacon_sim

namespace std {
template <>
struct std::hash<robot::experimental::beacon_sim::RobotBelief> {
    std::size_t operator()(const robot::experimental::beacon_sim::RobotBelief &belief) const {
        std::hash<double> double_hasher;
        // This is probably a terrible hash function
        return (double_hasher(belief.local_from_robot.log().norm()) << 3) ^
               double_hasher(belief.cov_in_robot.determinant());
    }
};
}  // namespace std
