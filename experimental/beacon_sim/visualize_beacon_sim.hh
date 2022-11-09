
#pragma once

#include "experimental/beacon_sim/beacon_sim_state.hh"

namespace robot::experimental::beacon_sim {
void visualize_beacon_sim(const BeaconSimState &state, const double zoom_factor,
                          const double window_aspect_ratio);
}  // namespace robot::experimental::beacon_sim
