
#pragma once

#include "domain/canadian_traveler.hh"

namespace robot::experimental::ctp {

struct Rollout {
    std::vector<int> path;
    double cost;
};

Rollout compute_optimistic_rollout(const domain::CanadianTravelerGraph &graph,
                                   const int initial_state, const int goal_state,
                                   const domain::CanadianTravelerGraph::Weather &weather);

}  // namespace robot::experimental::ctp
