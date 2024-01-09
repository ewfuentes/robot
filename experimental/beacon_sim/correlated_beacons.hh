
#include <vector>

#include "experimental/beacon_sim/beacon_potential.hh"

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

BeaconPotential create_correlated_beacons(const BeaconClique &clique);

}  // namespace robot::experimental::beacon_sim
