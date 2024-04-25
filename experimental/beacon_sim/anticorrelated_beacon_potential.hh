
#pragma once

#include <random>
#include <unordered_map>
#include <vector>

#include "common/argument_wrapper.hh"
#include "experimental/beacon_sim/log_marginal.hh"

namespace robot::experimental::beacon_sim {

struct AnticorrelatedBeaconPotential {
    std::vector<int> members;
};

double compute_log_prob(const AnticorrelatedBeaconPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignment);

const std::vector<int> &get_members(const AnticorrelatedBeaconPotential &pot);

std::vector<LogMarginal> compute_log_marginals(const AnticorrelatedBeaconPotential &pot,
                                               const std::vector<int> &remaining);

std::vector<int> generate_sample(const AnticorrelatedBeaconPotential &pot, InOut<std::mt19937> gen);

namespace proto {
class BeaconPotential;
}

void pack_into_potential(const AnticorrelatedBeaconPotential &in, proto::BeaconPotential *);
}  // namespace robot::experimental::beacon_sim
