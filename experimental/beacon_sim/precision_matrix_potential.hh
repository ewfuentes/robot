
#pragma once

#include <random>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "experimental/beacon_sim/log_marginal.hh"

namespace robot::experimental::beacon_sim {

struct PrecisionMatrixPotential {
    Eigen::MatrixXd precision;
    double log_normalizer;
    std::vector<int> members;
};

double compute_log_prob(const PrecisionMatrixPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignment);

const std::vector<int> &get_members(const PrecisionMatrixPotential &pot);

std::vector<LogMarginal> compute_log_marginals(const PrecisionMatrixPotential &pot,
                                               const std::vector<int> &remaining);

std::vector<int> generate_sample(const PrecisionMatrixPotential &pot, InOut<std::mt19937> gen);

namespace proto {
class BeaconPotential;
}
void pack_into_potential(const PrecisionMatrixPotential &in, proto::BeaconPotential *);

}  // namespace robot::experimental::beacon_sim
