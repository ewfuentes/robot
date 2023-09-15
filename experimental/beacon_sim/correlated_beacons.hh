
#pragma once

#include <unordered_map>
#include <vector>

#include "Eigen/Dense"

namespace robot::experimental::beacon_sim {

// A beacon clique represents a set of beacons that are correlated in their presence/absence.
struct BeaconClique {
    // The marginal probabilityof a single beacon within the clique
    double p_beacon;

    // The probability of no beacons appearing
    double p_no_beacons;

    // The beacon ids associated with this clique
    std::vector<int> members;
};

struct LogMarginal {
    std::vector<int> present_beacons;
    double log_marginal;
};

// A probability distribution over beacon presences/absences
class BeaconPotential {
   public:
    BeaconPotential() = default;
    BeaconPotential(const Eigen::MatrixXd &information, const double log_normalizer,
                    const std::vector<int> &members);

    double log_prob(const std::unordered_map<int, bool> &assignments) const;
    double log_prob(const std::vector<int> &present_beacons) const;

    BeaconPotential operator*(const BeaconPotential &other) const;

    const Eigen::MatrixXd &precision() const { return precision_; };
    double log_normalizer() const { return log_normalizer_; };

    std::vector<LogMarginal> compute_log_marginals(const std::vector<int> &remaining) const;

    const std::vector<int> &members() const { return members_; };

   private:
    Eigen::MatrixXd precision_;
    double log_normalizer_;
    std::vector<int> members_;
};

BeaconPotential create_correlated_beacons(const BeaconClique &clique);
}  // namespace robot::experimental::beacon_sim
