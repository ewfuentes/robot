
#include "experimental/beacon_sim/anticorrelated_beacon_potential.hh"

#include <algorithm>
#include <limits>

#include "experimental/beacon_sim/beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace {
std::vector<int> sorted_vector(const std::vector<int> &in) {
    std::vector<int> sorted_members = in;
    std::sort(sorted_members.begin(), sorted_members.end());
    return sorted_members;
}
}  // namespace

double compute_log_prob(const AnticorrelatedBeaconPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        [[maybe_unused]] const bool allow_partial_assignments) {
    int num_landmarks_present = 0;
    int num_landmarks_absent = 0;
    for (const auto &member : pot.members) {
        const auto iter = assignments.find(member);
        if (iter != assignments.end()) {
            num_landmarks_present += iter->second;
            num_landmarks_absent += !iter->second;
        }
    }

    const bool is_more_than_one_present = num_landmarks_present > 1;
    const bool are_all_absent = num_landmarks_absent == static_cast<int>(pot.members.size());
    if (is_more_than_one_present || are_all_absent) {
        return -std::numeric_limits<double>::infinity();
    } else if (num_landmarks_present == 1) {
        return std::log(1.0 / pot.members.size());
    } else {
        // This is the case where no landmarks are assigned present and
        // none or some of the landmarks are marked absent
        return std::log(static_cast<double>(pot.members.size() - num_landmarks_absent) /
                        pot.members.size());
    }
}

const std::vector<int> &get_members(const AnticorrelatedBeaconPotential &pot) {
    return pot.members;
}

std::vector<LogMarginal> compute_log_marginals(const AnticorrelatedBeaconPotential &pot,
                                               const std::vector<int> &all_remaining) {
    const auto sorted_members = sorted_vector(pot.members);
    const auto sorted_all_remaining = sorted_vector(all_remaining);

    std::vector<int> remaining;
    std::set_intersection(sorted_all_remaining.begin(), sorted_all_remaining.end(),
                          sorted_members.begin(), sorted_members.end(),
                          std::back_inserter(remaining));

    std::vector<LogMarginal> out;
    out.push_back(LogMarginal{
        .present_beacons = {},
        .log_marginal = std::log(static_cast<double>(pot.members.size() - remaining.size()) /
                                 pot.members.size())});

    for (const auto remaining_member : remaining) {
        out.push_back(LogMarginal{.present_beacons = {remaining_member},
                                  .log_marginal = std::log(1.0 / pot.members.size())});
    }
    return out;
}

std::vector<int> generate_sample(const AnticorrelatedBeaconPotential &pot,
                                 InOut<std::mt19937> gen) {
    std::uniform_int_distribution<> dist(0, pot.members.size() - 1);
    const int selected_idx = dist(*gen);
    return {pot.members.at(selected_idx)};
}

}  // namespace robot::experimental::beacon_sim
