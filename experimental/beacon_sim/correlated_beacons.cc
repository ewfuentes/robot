
#include "experimental/beacon_sim/correlated_beacons.hh"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <unordered_map>

#include "common/check.hh"
#include "common/math/n_choose_k.hh"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "experimental/beacon_sim/precision_matrix_potential.hh"

namespace robot::experimental::beacon_sim {

namespace {

auto logsumexp(const auto &terms) {
    using drake::symbolic::max;
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

drake::symbolic::Expression compute_marginal_log_prob(const int n,
                                                      const drake::symbolic::Variable &phi,
                                                      const drake::symbolic::Variable &psi,
                                                      const double bias) {
    const auto kth_term = [&phi, &psi, &bias](const int n,
                                              const int k) -> drake::symbolic::Expression {
        return std::log(static_cast<double>(math::n_choose_k(n, k))) +
               phi * static_cast<double>(k + 1) + static_cast<double>((k + 1) * k / 2) * psi - bias;
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
                                                   const double bias) {
    using drake::symbolic::operator*;
    const auto kth_term = [&phi, &psi, &bias](const int n,
                                              const int k) -> drake::symbolic::Expression {
        return std::log(static_cast<double>(math::n_choose_k(n, k))) +
               phi * static_cast<double>(k) + static_cast<double>((k - 1) * k / 2) * psi - bias;
    };

    std::vector<drake::symbolic::Expression> terms;
    terms.reserve(n);
    for (int k = 0; k < n + 1; k++) {
        terms.push_back(kth_term(n, k));
    }

    return logsumexp(terms);
}
}  // namespace

BeaconPotential create_correlated_beacons(const BeaconClique &clique) {
    drake::solvers::MathematicalProgram program;
    const auto precision_matrix_vars = program.NewContinuousVariables(2, "prec");
    const auto &diag = precision_matrix_vars[0];
    const auto &off_diag = precision_matrix_vars[1];
    const double log_norm = -std::log(clique.p_no_beacons);
    program.AddConstraint(compute_total_log_prob(clique.members.size(), diag, off_diag, log_norm) ==
                          0.0);
    program.AddConstraint(compute_marginal_log_prob(clique.members.size(), diag, off_diag,
                                                    log_norm) == std::log(clique.p_beacon));
    program.AddCost(diag * diag + off_diag * off_diag);
    // Set an initial guess away from 0.0. I'm not sure why, but it seems like the gradient is NaN
    // when all parameters equal 0.0.
    program.SetInitialGuess(diag, 1.0);
    program.SetInitialGuess(off_diag, 1.0);

    const auto result = Solve(program);

    const int n = clique.members.size();
    Eigen::MatrixXd precision(n, n);
    for (int i = 0; i < n; i++) {
        precision(i, i) = result.GetSolution(diag);
        for (int j = i + 1; j < n; j++) {
            precision(i, j) = result.GetSolution(off_diag) / 2.0;
            precision(j, i) = result.GetSolution(off_diag) / 2.0;
        }
    }

    return PrecisionMatrixPotential{
        .precision = precision, .log_normalizer = log_norm, .members = clique.members};
}
}  // namespace robot::experimental::beacon_sim
