
#pragma once

#include <unordered_map>
#include <vector>

#include "experimental/beacon_sim/beacon_potential.hh"

namespace robot::experimental::beacon_sim {
namespace proto {
class BeaconPotential;
}

// A ConditionedPotential is meant to represent a BeaconPotential where
// certain beacons are assumed to take a given value. A conditioned potential
// maintains the same set of members as the underlying distribution, but
// if compute_log_prob is queried with a conflicting assignment, a log probability
// of -infinity is returned. `compute_log_marginals` will contain the unconditioned
// variables and the conditioned variables with the given assignment.
struct ConditionedPotential {
    BeaconPotential underlying_pot;
    // The probability of the conditioned assignment on `underlying_pot`
    double log_normalizer;
    std::unordered_map<int, bool> conditioned_members;
};

double compute_log_prob(const ConditionedPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignments);

std::vector<LogMarginal> compute_log_marginals(const ConditionedPotential &pot,
                                               const std::vector<int> &remaining);
std::vector<int> get_members(const ConditionedPotential &pot);
void pack_into_potential(const ConditionedPotential &in, proto::BeaconPotential *out);
std::vector<int> generate_sample(const ConditionedPotential &pot, InOut<std::mt19937> gen);
}  // namespace robot::experimental::beacon_sim
