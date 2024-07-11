
#pragma once

#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "common/argument_wrapper.hh"
#include "experimental/beacon_sim/log_marginal.hh"

namespace robot::experimental::beacon_sim {

// A correlated beacon potential models the following random variable:
// Consider a Bernoulli random variable X with parameter `p_present`.
// If X = 0, then no beacons are present. If X = 1, then presence of the
// ith landmark is modeled by a Bernoulli random variable Y_i with parameter
// `p_beacon_given_present`.
struct CorrelatedBeaconPotential {
    // The probability of the first coin flip
    double p_present;
    // The probability of a single beacon the first coin flip is successful
    double p_beacon_given_present;
    // The beacon ids associated with this clique
    std::vector<int> members;

    struct ConditioningBlock {
        std::unordered_map<int, bool> conditioned_members;
    };

    std::optional<ConditioningBlock> conditioning = std::nullopt;
};

double compute_log_prob(const CorrelatedBeaconPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignment);

const std::vector<int> &get_members(const CorrelatedBeaconPotential &pot);

std::vector<LogMarginal> compute_log_marginals(const CorrelatedBeaconPotential &pot,
                                               const std::vector<int> &remaining);

std::vector<int> generate_sample(const CorrelatedBeaconPotential &pot, InOut<std::mt19937> gen);

CorrelatedBeaconPotential condition_on(const CorrelatedBeaconPotential &pot,
                                       const std::unordered_map<int, bool> &assignment);
void recondition_on(CorrelatedBeaconPotential &pot,
                    const std::unordered_map<int, bool> &assignment);

namespace proto {
class BeaconPotential;
}
void pack_into_potential(const CorrelatedBeaconPotential &in, proto::BeaconPotential *);

}  // namespace robot::experimental::beacon_sim
