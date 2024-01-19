
#include "experimental/beacon_sim/conditioned_potential.hh"

#include <limits>
#include <unordered_set>

#include "common/math/sample_without_replacement.hh"

namespace robot::experimental::beacon_sim {
double compute_log_prob(const ConditionedPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignments) {
    auto query = pot.conditioned_members;
    for (const auto &[beacon_id, is_present] : assignments) {
        const auto iter = pot.conditioned_members.find(beacon_id);
        if (iter != pot.conditioned_members.end() && iter->second != is_present) {
            return -std::numeric_limits<double>::infinity();
        }
        query[beacon_id] = is_present;
    }
    return pot.underlying_pot.log_prob(query, allow_partial_assignments) - pot.log_normalizer;
}

std::vector<LogMarginal> compute_log_marginals(const ConditionedPotential &pot,
                                               const std::vector<int> &remaining) {
    // Check that the remaining members are consistent with the conditioned members
    std::vector<int> all_remaining = remaining;
    for (const auto &[beacon_id, is_present] : pot.conditioned_members) {
        if (is_present) {
            all_remaining.push_back(beacon_id);
        } else {
            const auto iter = std::find(remaining.begin(), remaining.end(), beacon_id);
            if (iter != remaining.end()) {
                // We found a beacon id present in remaining that should be missing
                // according to the conditioning. Return an empty set of marginals
                return {};
            }
        }
    }

    // Generate the log marginals from the underlying distribution and filter out the
    // ones that aren't consistent
    auto log_marginals = pot.underlying_pot.log_marginals(all_remaining);

    std::erase_if(log_marginals, [&](const auto &log_marginal) {
        const std::unordered_set<int> present_set(log_marginal.present_beacons.begin(),
                                                  log_marginal.present_beacons.end());

        for (const auto &[beacon_id, is_present] : pot.conditioned_members) {
            if (present_set.contains(beacon_id) != is_present) {
                return true;
            }
        }
        return false;
    });

    // Adjust the probabilities for the conditioning
    for (auto &log_marginal : log_marginals) {
        log_marginal.log_marginal -= pot.log_normalizer;
    }

    return log_marginals;
}

std::vector<int> get_members(const ConditionedPotential &pot) {
    return pot.underlying_pot.members();
}

std::vector<int> generate_sample(const ConditionedPotential &pot, InOut<std::mt19937> gen) {
    // We sample by computing the log marginals, and then sampling from these. There are
    // other ways (e.g. rejection sampling, MCMC) that may lead to more efficient sampling
    // but this is the easiest to implement at the moment.

    const auto marginals = compute_log_marginals(pot, get_members(pot));
    std::vector<double> log_p;
    std::transform(marginals.begin(), marginals.end(), std::back_inserter(log_p),
                   [](const auto &marginal) { return marginal.log_marginal; });

    constexpr int NUM_SAMPLES = 1;
    constexpr bool LOG_PROB = true;
    const auto sample_idx = math::sample_without_replacement(log_p, NUM_SAMPLES, LOG_PROB, gen);

    return marginals.at(sample_idx.at(0)).present_beacons;
}
}  // namespace robot::experimental::beacon_sim
