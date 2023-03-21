
#include "experimental/beacon_sim/correlated_beacons.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "common/math/n_choose_k.hh"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace robot::experimental::beacon_sim {

namespace {
auto logsumexp(const auto &terms) {
    if (terms.size() == 1) {
        return terms[0];
    }
    const auto max_elem =
        std::accumulate(terms.begin() + 1, terms.end(), *terms.begin(), drake::symbolic::max);

    const auto sum = std::accumulate(
        terms.begin() + 1, terms.end(), exp(terms[0] - max_elem),
        [&max_elem](const auto &accum, const auto &term) { return accum + exp(term - max_elem); });

    return log(sum) + max_elem;
}

drake::symbolic::Expression compute_marginal_log_prob(const int n,
                                                      const drake::symbolic::Variable &phi,
                                                      const drake::symbolic::Variable &psi,
                                                      const drake::symbolic::Variable &bias) {
    const auto kth_term = [&phi, &psi, &bias](const int n,
                                              const int k) -> drake::symbolic::Expression {
        return std::log(static_cast<double>(math::n_choose_k(n, k))) +
               static_cast<double>(k + 1) * phi + static_cast<double>((k + 1) * k / 2) * psi + bias;
    };
    std::vector<drake::symbolic::Expression> terms;
    terms.reserve(n);
    for (int k = 0; k < n; k++) {
        terms.push_back(kth_term(n - 1, k));
    }

    return logsumexp(terms);
}

drake::symbolic::Expression compute_total_log_prob(const int n,
                                                   const drake::symbolic::Variable &phi,
                                                   const drake::symbolic::Variable &psi,
                                                   const drake::symbolic::Variable &bias) {
    const auto kth_term = [&phi, &psi, &bias](const int n,
                                              const int k) -> drake::symbolic::Expression {
        return std::log(static_cast<double>(math::n_choose_k(n, k))) +
               static_cast<double>(k) * phi + static_cast<double>((k - 1) * k / 2) * psi + bias;
    };

    std::vector<drake::symbolic::Expression> terms;
    terms.reserve(n);
    for (int k = 0; k < n + 1; k++) {
        terms.push_back(kth_term(n, k));
    }

    return logsumexp(terms);
}
}  // namespace

BeaconPotential::BeaconPotential(const Eigen::MatrixXd &covariance, const double bias,
                                 const std::vector<int> &members)
    : covariance_(covariance), bias_(bias), members_(members) {}

BeaconPotential BeaconPotential::operator*(const BeaconPotential &other) {
    const int my_size = members_.size();
    const int other_size = other.members_.size();
    const int new_size = my_size + other_size;
    Eigen::MatrixXd new_cov = Eigen::MatrixXd::Zero(new_size, new_size);
    new_cov.topLeftCorner(my_size, my_size) = covariance_;
    new_cov.bottomRightCorner(other_size, other_size) = other.covariance_;
    const double new_bias = bias_ + other.bias_;
    std::vector<int> new_members(members_.begin(), members_.end());
    new_members.insert(new_members.end(), other.members_.begin(), other.members_.end());
    return BeaconPotential(new_cov, new_bias, new_members);
}

double BeaconPotential::log_prob(const std::unordered_map<int, bool> &assignment) const {
    Eigen::VectorXd x(members_.size());
    for (int i = 0; i < static_cast<int>(members_.size()); i++) {
        x(i) = assignment.at(members_.at(i));
    }
    return x.transpose() * covariance_ * x + bias_;
}

BeaconPotential create_correlated_beacons(const BeaconClique &clique) {
    drake::solvers::MathematicalProgram program;
    const auto covariance_matrix_vars = program.NewContinuousVariables(2, "covar");
    const auto &diag = covariance_matrix_vars[0];
    const auto &off_diag = covariance_matrix_vars[1];
    const auto bias_var = program.NewContinuousVariables(1, "offset");
    const auto &bias = bias_var[0];
    program.AddLinearEqualityConstraint(bias, std::log(clique.p_no_beacons));
    program.AddConstraint(compute_total_log_prob(clique.members.size(), diag, off_diag, bias) ==
                          0.0);
    program.AddConstraint(compute_marginal_log_prob(clique.members.size(), diag, off_diag, bias) ==
                          std::log(clique.p_beacon));
    program.AddCost(diag * diag + off_diag * off_diag + bias * bias);
    // Set an initial guess away from 0.0. I'm not sure why, but it seems like the gradient is NaN
    // when all parameters equal 0.0.
    program.SetInitialGuess(diag, 1.0);
    program.SetInitialGuess(off_diag, 1.0);
    program.SetInitialGuess(bias, 1.0);

    const auto result = Solve(program);

    const int n = clique.members.size();
    Eigen::MatrixXd cov(n, n);
    for (int i = 0; i < n; i++) {
        cov(i, i) = result.GetSolution(diag);
        for (int j = i + 1; j < n; j++) {
            cov(i, j) = result.GetSolution(off_diag) / 2.0;
            cov(j, i) = result.GetSolution(off_diag) / 2.0;
        }
    }

    return BeaconPotential(cov, result.GetSolution(bias), clique.members);
}
}  // namespace robot::experimental::beacon_sim
