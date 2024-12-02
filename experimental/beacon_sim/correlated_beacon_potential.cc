
#include "experimental/beacon_sim/correlated_beacon_potential.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#include "common/check.hh"
#include "common/math/combinations.hh"
#include "common/math/n_choose_k.hh"

namespace robot::experimental::beacon_sim {
namespace {

auto logsumexp(const auto &terms) {
    using std::max;
    using T = typename std::decay_t<decltype(terms)>::value_type;
    if (terms.size() == 1) {
        return terms[0];
    }

    const auto max_elem = std::accumulate(terms.begin() + 1, terms.end(), *terms.begin(),
                                          [](const T &a, const T &b) { return max(a, b); });

    const auto sum = std::accumulate(
        terms.begin() + 1, terms.end(), exp(terms[0] - max_elem),
        [&max_elem](const auto &accum, const auto &term) { return accum + exp(term - max_elem); });

    return log(sum) + max_elem;
}

std::vector<int> sorted_vector(const std::vector<int> &in) {
    std::vector<int> sorted_members = in;
    std::sort(sorted_members.begin(), sorted_members.end());
    return sorted_members;
}

std::vector<int> sorted_keys(const std::unordered_map<int, bool> &in) {
    std::vector<int> items;
    std::transform(in.begin(), in.end(), std::back_inserter(items),
                   [](const auto &key_and_value) { return key_and_value.first; });
    std::sort(items.begin(), items.end());
    return items;
}
}  // namespace

double compute_log_prob(const CorrelatedBeaconPotential &pot,
                        const std::unordered_map<int, bool> &assignment,
                        const bool allow_partial_assignment) {
    const std::vector<int> sorted_members = sorted_vector(pot.members);
    const std::vector<int> keys = sorted_keys(assignment);
    const std::vector<int> conditioned_keys =
        pot.conditioning.has_value() ? sorted_keys(pot.conditioning->conditioned_members)
                                     : std::vector<int>{};

    // Check to see if the assignment is in conflict with the conditioning
    for (const int key : conditioned_keys) {
        const auto iter = assignment.find(key);
        if (iter != assignment.end() &&
            iter->second != pot.conditioning->conditioned_members.at(key)) {
            return -std::numeric_limits<double>::infinity();
        }
    }

    std::vector<int> missing_keys;
    std::set_difference(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                        std::back_inserter(missing_keys));

    std::vector<int> keys_to_keep;
    std::set_intersection(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                          std::back_inserter(keys_to_keep));

    ROBOT_CHECK(allow_partial_assignment || missing_keys.empty(),
          "partial assignment specified when not enabled", assignment, missing_keys, pot.members);

    const int num_beacons_present =
        std::count_if(keys_to_keep.begin(), keys_to_keep.end(),
                      [&assignment](const int beacon_id) { return assignment.at(beacon_id); });
    const int num_beacons_missing = keys_to_keep.size() - num_beacons_present;

    const int num_conditioned_present = std::count_if(
        conditioned_keys.begin(), conditioned_keys.end(), [&pot](const int beacon_id) {
            return pot.conditioning->conditioned_members.at(beacon_id);
        });

    const int num_conditioned_present_specified = std::count_if(
        conditioned_keys.begin(), conditioned_keys.end(), [&pot, &assignment](const int beacon_id) {
            return pot.conditioning->conditioned_members.at(beacon_id) &&
                   assignment.contains(beacon_id);
        });
    const int num_conditioned_absent_specified = std::count_if(
        conditioned_keys.begin(), conditioned_keys.end(), [&pot, &assignment](const int beacon_id) {
            return !pot.conditioning->conditioned_members.at(beacon_id) &&
                   assignment.contains(beacon_id);
        });

    const int num_unconditioned_present = num_beacons_present - num_conditioned_present_specified;
    const int num_unconditioned_absent = num_beacons_missing - num_conditioned_absent_specified;

    std::vector<double> terms;
    if (num_beacons_present + num_conditioned_present == 0) {
        terms.push_back(std::log(1 - pot.p_present));
    }

    terms.push_back(num_unconditioned_present * std::log(pot.p_beacon_given_present) +
                    num_unconditioned_absent * std::log(1 - pot.p_beacon_given_present) +
                    (num_conditioned_present == 0 ? std::log(pot.p_present) : 0.0));

    return logsumexp(terms);
}

const std::vector<int> &get_members(const CorrelatedBeaconPotential &pot) { return pot.members; }

std::vector<LogMarginal> compute_log_marginals(const CorrelatedBeaconPotential &pot,
                                               const std::vector<int> &all_remaining) {
    const auto sorted_all_remaining = sorted_vector(all_remaining);
    const auto sorted_pot_members = sorted_vector(pot.members);
    const std::vector<int> conditioned_keys =
        pot.conditioning.has_value() ? sorted_keys(pot.conditioning->conditioned_members)
                                     : std::vector<int>{};
    std::vector<int> remaining;
    std::set_intersection(sorted_all_remaining.begin(), sorted_all_remaining.end(),
                          sorted_pot_members.begin(), sorted_pot_members.end(),
                          std::back_inserter(remaining));

    std::vector<int> unconditioned_remaining;
    std::set_difference(remaining.begin(), remaining.end(), conditioned_keys.begin(),
                        conditioned_keys.end(), std::back_inserter(unconditioned_remaining));

    const int n_remaining = unconditioned_remaining.size();
    std::vector<LogMarginal> out;
    out.reserve(1 << n_remaining);
    std::unordered_map<int, bool> assignment = pot.conditioning.has_value()
                                                   ? pot.conditioning->conditioned_members
                                                   : std::unordered_map<int, bool>{};
    std::vector<int> present_conditioned;
    if (pot.conditioning.has_value()) {
        for (const auto &[id, value] : pot.conditioning->conditioned_members) {
            if (value) {
                present_conditioned.push_back(id);
            }
        }
    }
    for (int num_present = 0; num_present <= n_remaining; num_present++) {
        for (const auto &config : math::combinations(n_remaining, num_present)) {
            for (const int present_beacon_id : unconditioned_remaining) {
                assignment[present_beacon_id] = false;
            }
            std::vector<int> present_beacons = present_conditioned;
            for (const int remaining_idx : config) {
                const int present_id = unconditioned_remaining.at(remaining_idx);
                present_beacons.push_back(present_id);
                assignment[present_id] = true;
            }

            constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
            out.emplace_back(LogMarginal{
                .present_beacons = std::move(present_beacons),
                .log_marginal = compute_log_prob(pot, assignment, ALLOW_PARTIAL_ASSIGNMENT),
            });
        }
    }
    return out;
}

std::vector<int> generate_sample(const CorrelatedBeaconPotential &pot, InOut<std::mt19937> gen) {
    const std::vector<int> sorted_conditioned =
        pot.conditioning.has_value() ? sorted_keys(pot.conditioning->conditioned_members)
                                     : std::vector<int>{};
    std::vector<int> present_conditioned;
    if (pot.conditioning.has_value()) {
        for (const auto &[id, value] : pot.conditioning->conditioned_members) {
            if (value) {
                present_conditioned.push_back(id);
            }
        }
    }
    present_conditioned = sorted_vector(present_conditioned);

    std::uniform_real_distribution<> dist;
    const bool are_beacon_present = dist(*gen) < pot.p_present || (present_conditioned.size() > 0);
    if (!are_beacon_present) {
        return {};
    }

    std::vector<int> sorted_pot_members = sorted_vector(pot.members);
    std::vector<int> unconditioned_members;
    std::set_difference(sorted_pot_members.begin(), sorted_pot_members.end(),
                        sorted_conditioned.begin(), sorted_conditioned.end(),
                        std::back_inserter(unconditioned_members));
    std::vector<int> out = std::move(present_conditioned);
    // Given the flip above is successful, then the landmarks are independent
    for (const int beacon_id : unconditioned_members) {
        if (dist(*gen) < pot.p_beacon_given_present) {
            out.push_back(beacon_id);
        }
    }
    return out;
}

CorrelatedBeaconPotential condition_on(const CorrelatedBeaconPotential &pot,
                                       const std::unordered_map<int, bool> &assignment) {
    std::unordered_map<int, bool> new_assignment;
    if (pot.conditioning.has_value()) {
        // Need to ensure that the additional assigments are consistent with the existing
        // assignments
        new_assignment = pot.conditioning->conditioned_members;
        for (const auto &[id, value] : assignment) {
            const auto &existing_conditioned = pot.conditioning->conditioned_members;
            ROBOT_CHECK(!existing_conditioned.contains(id) || (existing_conditioned.at(id) == value),
                  "Inconsistent conditioning", existing_conditioned, assignment);

            new_assignment[id] = value;
        }
    } else {
        new_assignment = assignment;
    }

    return CorrelatedBeaconPotential{
        .p_present = pot.p_present,
        .p_beacon_given_present = pot.p_beacon_given_present,
        .members = pot.members,
        .conditioning = new_assignment.empty()
                            ? std::nullopt
                            : std::make_optional(CorrelatedBeaconPotential::ConditioningBlock{
                                  .conditioned_members = new_assignment})};
}

void recondition_on(CorrelatedBeaconPotential &pot,
                    const std::unordered_map<int, bool> &assignment) {
    pot.conditioning = {.conditioned_members = assignment};
}

}  // namespace robot::experimental::beacon_sim
