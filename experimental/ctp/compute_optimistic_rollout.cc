
#include "experimental/ctp/compute_optimistic_rollout.hh"

#include "domain/canadian_traveler.hh"

namespace robot::experimental::ctp {
namespace {
using CTG = domain::CanadianTravelerGraph;
}

Rollout compute_optimistic_rollout([[maybe_unused]] const CTG &graph, const int initial_state,
                                   const int goal_state,
                                   [[maybe_unused]] const CTG::Weather &underlying_weather,
                                   [[maybe_unused]] const CTG::Weather &initial_belief) {
    Rollout out = {
        .path = {initial_state},
        .cost = 0.0,
    };

    while (out.path.back() != goal_state) {
        // Observe at the current node given the underlying weather
        // Create an optimistic belief
        // Run Djikstra towards to goal assuming the belief
        // Step in the best direction
    }

    return out;
}
}  // namespace robot::experimental::ctp
