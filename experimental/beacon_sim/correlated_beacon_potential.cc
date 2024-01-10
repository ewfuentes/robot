
#include "experimental/beacon_sim/correlated_beacon_potential.hh"

#include <algorithm>
#include <cmath>
#include <numeric>

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
    std::vector<int> missing_keys;
    std::set_difference(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                        std::back_inserter(missing_keys));

    std::vector<int> keys_to_keep;
    std::set_intersection(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                          std::back_inserter(keys_to_keep));

    CHECK(allow_partial_assignment || missing_keys.empty(),
          "partial assignment specified when not enabled", assignment, missing_keys, pot.members);

    const int num_beacons_present =
        std::count_if(keys_to_keep.begin(), keys_to_keep.end(),
                      [&assignment](const int beacon_id) { return assignment.at(beacon_id); });
    const int num_beacons_missing = keys_to_keep.size() - num_beacons_present;

    std::vector<double> terms;

    if (num_beacons_present == 0) {
        terms.push_back(std::log(1 - pot.p_present));
    }

    terms.push_back(num_beacons_present * std::log(pot.p_beacon_given_present) +
                    num_beacons_missing * std::log(1 - pot.p_beacon_given_present) +
                    std::log(pot.p_present));

    return logsumexp(terms);
}

double compute_log_prob(const CorrelatedBeaconPotential &pot,
                        const std::vector<int> &present_beacons) {
    std::unordered_map<int, bool> assignment;
    for (const int beacon_id : pot.members) {
        const bool is_beacon_present = std::find(present_beacons.begin(), present_beacons.end(),
                                                 beacon_id) != present_beacons.end();
        assignment[beacon_id] = is_beacon_present;
    }

    constexpr bool DONT_ALLOW_PARTIAL_ASSIGNMENT = false;
    return compute_log_prob(pot, assignment, DONT_ALLOW_PARTIAL_ASSIGNMENT);
}

std::vector<int> get_members(const CorrelatedBeaconPotential &pot) { return pot.members; }

std::vector<LogMarginal> compute_log_marginals(const CorrelatedBeaconPotential &pot,
                                               const std::vector<int> &all_remaining) {
    const auto sorted_all_remaining = sorted_vector(all_remaining);
    const auto sorted_pot_members = sorted_vector(pot.members);
    std::vector<int> remaining;
    std::set_intersection(sorted_all_remaining.begin(), sorted_all_remaining.end(),
                          sorted_pot_members.begin(), sorted_pot_members.end(),
                          std::back_inserter(remaining));

    const int n_remaining = remaining.size();
    std::vector<LogMarginal> out;
    out.reserve(1 << n_remaining);
    std::unordered_map<int, bool> assignment;
    for (int num_present = 0; num_present <= n_remaining; num_present++) {
        for (const auto &config : math::combinations(n_remaining, num_present)) {
            for (const int present_beacon_id : remaining) {
                assignment[present_beacon_id] = false;
            }
            std::vector<int> present_beacons;
            for (const int remaining_idx : config) {
                const int present_id = remaining.at(remaining_idx);
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
}  // namespace robot::experimental::beacon_sim
