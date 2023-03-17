
#include "experimental/beacon_sim/correlated_beacons.hh"

#include <unordered_map>

#include "drake/solvers/mathematical_program.h"

namespace robot::experimental::beacon_sim {

BeaconPotential::BeaconPotential(const Eigen::MatrixXd &covariance, const double bias,
                                 const std::vector<int> &members)
    : covariance_(covariance), bias_(bias), members_(members) {
  (void)covariance_;
  (void)bias_;
  (void)members_;
}

  BeaconPotential BeaconPotential::operator*(const BeaconPotential &) {return *this; }

double BeaconPotential::log_prob(const std::unordered_map<int, bool> &) const { return 0.0; }

BeaconPotential create_correlated_beacons(const BeaconClique &) {
    drake::solvers::MathematicalProgram program;
    return BeaconPotential(Eigen::MatrixXd(), 0.0, {});
}

}  // namespace robot::experimental::beacon_sim
