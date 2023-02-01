
#pragma once

#include "common/argument_wrapper.hh"
#include "experimental/beacon_sim/beacon_sim_debug.pb.h"
#include "experimental/beacon_sim/beacon_sim_state.hh"
#include "experimental/beacon_sim/sim_config.hh"

namespace robot::experimental::beacon_sim {
struct RobotCommand {
    double turn_rad;
    double move_m;
};

proto::BeaconSimDebug tick_sim(const SimConfig &sim_config, const RobotCommand &command,
                               InOut<BeaconSimState> state);
}  // namespace robot::experimental::beacon_sim
