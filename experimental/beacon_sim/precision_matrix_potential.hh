
#pragma once

#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "experimental/beacon_sim/beacon_potential.hh"

namespace robot::experimental::beacon_sim {

struct PrecisionMatrixPotential {
    Eigen::MatrixXd precision;
    double log_normalizer;
    std::vector<int> members;
};

double compute_log_prob(const PrecisionMatrixPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignment);

double compute_log_prob(const PrecisionMatrixPotential &pot,
                        const std::vector<int> &present_beacons);

std::vector<int> get_members(const PrecisionMatrixPotential &pot);

std::vector<LogMarginal> compute_log_marginals(const PrecisionMatrixPotential &pot,
                                               const std::vector<int> &remaining);

namespace proto {
class BeaconPotential;
}
void pack_into_potential(const PrecisionMatrixPotential &in, proto::BeaconPotential *);

}  // namespace robot::experimental::beacon_sim
