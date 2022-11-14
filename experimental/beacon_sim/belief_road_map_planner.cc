
#include "experimental/beacon_sim/belief_road_map_planner.hh"

namespace robot::experimental::beacon_sim {
planning::BRMPlan compute_belief_road_map_plan(const planning::RoadMap &road_map,
                                               const EkfSlam &ekf) {
    (void)road_map;
    (void)ekf;
    return {};
}
}  // namespace robot::experimental::beacon_sim
