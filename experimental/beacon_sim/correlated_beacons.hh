
#include "Eigen/Dense"
#include <unordered_map>
#include <vector>

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

// A probability distribution over beacon presences/absences
class BeaconPotential {
   public:
    BeaconPotential(const Eigen::MatrixXd &covariance, const double bias,
                    const std::vector<int> &members);

    double log_prob(const std::unordered_map<int, bool> &assignments) const;

    BeaconPotential operator*(const BeaconPotential &other);

   private:
    Eigen::MatrixXd covariance_;
    double bias_;
    std::vector<int> members_;
};

BeaconPotential create_correlated_beacons(const BeaconClique &clique);
}  // namespace robot::experimental::beacon_sim
