
#include "experimental/beacon_sim/beacon_potential.hh"

#include <algorithm>
#include <iostream>
#include <numeric>

namespace robot::experimental::beacon_sim {

CombinedPotential::CombinedPotential(std::vector<BeaconPotential> pots_) : pots(std::move(pots_)) {
    for (const BeaconPotential &p : pots) {
        const auto p_members = p.members();
        members.insert(members.end(), p_members.begin(), p_members.end());
    }
}

double compute_log_prob(const CombinedPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignments) {
    return std::accumulate(
        pot.pots.begin(), pot.pots.end(), 0.0,
        [&assignments, allow_partial_assignments](const double accum, const BeaconPotential &pot) {
            return accum + pot.log_prob(assignments, allow_partial_assignments);
        });
}

std::vector<LogMarginal> compute_log_marginals(const CombinedPotential &pot,
                                               const std::vector<int> &remaining) {
    std::vector<std::vector<LogMarginal>> log_marginals;
    std::transform(
        pot.pots.begin(), pot.pots.end(), std::back_inserter(log_marginals),
        [&remaining](const BeaconPotential &pot) { return pot.log_marginals(remaining); });

    std::vector<LogMarginal> out;
    std::vector<int> idxs(pot.pots.size(), 0);
    bool run = true;
    while (run) {
        std::vector<int> present_beacons;
        double log_marginal = 0;

        for (int pot_idx = 0; pot_idx < static_cast<int>(idxs.size()); pot_idx++) {
            const int idx_for_pot = idxs.at(pot_idx);
            const auto &pot_marginal = log_marginals.at(pot_idx).at(idx_for_pot);
            present_beacons.insert(present_beacons.end(), pot_marginal.present_beacons.begin(),
                                   pot_marginal.present_beacons.end());

            log_marginal += pot_marginal.log_marginal;
        }

        out.emplace_back(LogMarginal{.present_beacons = std::move(present_beacons),
                                     .log_marginal = log_marginal});

        // Update the state
        for (int pot_idx = 0; pot_idx < static_cast<int>(idxs.size()); pot_idx++) {
            idxs.at(pot_idx)++;
            if (idxs.at(pot_idx) == static_cast<int>(log_marginals.at(pot_idx).size())) {
                idxs.at(pot_idx) = 0;
            } else {
                break;
            }
        }

        // check if we should break
        // Since the state is updated before we check, the indices will all be equal to zero
        // only if we've iterated through all options
        run = std::any_of(idxs.begin(), idxs.end(), [](const int idx) { return idx > 0; });
    }
    return out;
}

const std::vector<int> &get_members(const CombinedPotential &pot) { return pot.members; }

std::vector<int> generate_sample(const CombinedPotential &pot, InOut<std::mt19937> gen) {
    std::vector<int> out;
    for (const auto &sub_pot : pot.pots) {
        const auto subsample = sub_pot.sample(gen);
        out.insert(out.end(), subsample.begin(), subsample.end());
    }
    return out;
}

CombinedPotential condition_on(const CombinedPotential &pot,
                               const std::unordered_map<int, bool> &assignments) {
    std::vector<BeaconPotential> conditioned_pots;
    for (const BeaconPotential &p : pot.pots) {
        std::unordered_map<int, bool> pot_assignments;
        for (const auto member : p.members()) {
            const auto iter = assignments.find(member);
            if (iter != assignments.end()) {
                pot_assignments.insert(*iter);
            }
        }

        conditioned_pots.emplace_back(p.conditioned_on(pot_assignments));
    }
    return CombinedPotential(std::move(conditioned_pots));
}

void recondition_on(CombinedPotential &pot, const std::unordered_map<int, bool> &assignments) {
    for (BeaconPotential &p : pot.pots) {
        std::unordered_map<int, bool> pot_assignments;
        for (const auto member : p.members()) {
            const auto iter = assignments.find(member);
            if (iter != assignments.end()) {
                pot_assignments.insert(*iter);
            }
        }
        p.reconditioned_on(pot_assignments);
    }
}

BeaconPotential operator*(const BeaconPotential &a, const BeaconPotential &b) {
    const auto sorted_vector = [](const std::vector<int> &in) {
        std::vector<int> sorted_members = in;
        std::sort(sorted_members.begin(), sorted_members.end());
        return sorted_members;
    };

    const auto a_sorted_members = sorted_vector(a.members());
    const auto b_sorted_members = sorted_vector(b.members());

    std::vector<int> common_elements;
    std::set_intersection(a_sorted_members.begin(), a_sorted_members.end(),
                          b_sorted_members.begin(), b_sorted_members.end(),
                          std::back_inserter(common_elements));
    CHECK(common_elements.empty(), "Found overlap in members of potentials", a_sorted_members,
          b_sorted_members, common_elements);

    return CombinedPotential({a, b});
}
}  // namespace robot::experimental::beacon_sim
