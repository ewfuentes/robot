
#include "experimental/beacon_sim/correlated_beacons.hh"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "common/math/combinations.hh"
#include "common/math/n_choose_k.hh"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace robot::experimental::beacon_sim {

namespace {

std::vector<int> sorted_vector(const std::vector<int> &in) {
    std::vector<int> sorted_members = in;
    std::sort(sorted_members.begin(), sorted_members.end());
    return sorted_members;
}

std::vector<int> sorted_keys(const std::unordered_map<int, bool> &in) {
    std::vector<int> items;
    std::transform(in.begin(), in.end(), std::back_inserter(items),
                   [](const auto &key_and_value) { return key_and_value.first; });
    std::sort(items.begin(), items.end());
    return items;
}

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
                                                      const drake::symbolic::Variable &bias) {
    const auto kth_term = [&phi, &psi, &bias](const int n,
                                              const int k) -> drake::symbolic::Expression {
        return std::log(static_cast<double>(math::n_choose_k(n, k))) +
               static_cast<double>(k + 1) * phi + static_cast<double>((k + 1) * k / 2) * psi - bias;
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
               static_cast<double>(k) * phi + static_cast<double>((k - 1) * k / 2) * psi - bias;
    };

    std::vector<drake::symbolic::Expression> terms;
    terms.reserve(n);
    for (int k = 0; k < n + 1; k++) {
        terms.push_back(kth_term(n, k));
    }

    return logsumexp(terms);
}
}  // namespace

BeaconPotential::BeaconPotential(const Eigen::MatrixXd &precision, const double log_norm,
                                 const std::vector<int> &members)
    : precision_(precision), log_normalizer_(log_norm), members_(members) {}

BeaconPotential BeaconPotential::operator*(const BeaconPotential &other) {
    const std::vector<int> sorted_members = sorted_vector(members_);
    const std::vector<int> other_sorted_members = sorted_vector(other.members_);

    std::vector<int> common_elements;
    std::set_intersection(sorted_members.begin(), sorted_members.end(),
                          other_sorted_members.begin(), other_sorted_members.end(),
                          std::back_inserter(common_elements));

    if (!common_elements.empty()) {
        std::ostringstream out;
        out << "BeaconPotential::operator* cannot accept arguments with members common elements: {";
        bool is_first = true;
        for (const auto &key : common_elements) {
            if (is_first) {
                is_first = false;
            } else {
                out << ", ";
            }
            out << key;
        }
        out << "}";
        throw std::runtime_error(out.str());
    }

    const int my_size = members_.size();
    const int other_size = other.members_.size();
    const int new_size = my_size + other_size;
    // Compute the new precision matrix
    Eigen::MatrixXd new_precision = Eigen::MatrixXd::Zero(new_size, new_size);
    new_precision.topLeftCorner(my_size, my_size) = precision_;
    new_precision.bottomRightCorner(other_size, other_size) = other.precision_;

    // Compute the new log normalizer
    const double new_log_norm = log_normalizer_ + other.log_normalizer_;

    std::vector<int> new_members(members_.begin(), members_.end());
    new_members.insert(new_members.end(), other.members_.begin(), other.members_.end());
    return BeaconPotential(new_precision, new_log_norm, new_members);
}

double BeaconPotential::log_prob(const std::unordered_map<int, bool> &assignment) const {
    const std::vector<int> sorted_members = sorted_vector(members_);
    const std::vector<int> keys = sorted_keys(assignment);
    std::vector<int> missing_keys;
    std::set_difference(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                        std::back_inserter(missing_keys));

    if (!missing_keys.empty()) {
        std::ostringstream out;
        out << "Missing keys from assignment {";
        bool is_first = true;
        for (const auto &key : missing_keys) {
            if (is_first) {
                is_first = false;
            } else {
                out << ", ";
            }
            out << key;
        }
        out << "}";
        throw std::runtime_error(out.str());
    }

    Eigen::VectorXd x(members_.size());
    for (int i = 0; i < static_cast<int>(members_.size()); i++) {
        x(i) = assignment.at(members_.at(i));
    }
    return x.transpose() * precision_ * x - log_normalizer_;
}

std::vector<LogMarginal> BeaconPotential::compute_log_marginals(const std::vector<int> &remaining) {
    // Find the members that we need to marginalize over
    const std::vector<int> to_marginalize = [&]() {
        std::vector<int> out;
        std::copy_if(members_.begin(), members_.end(), std::back_inserter(out),
                     [&](const int &member) {
                         const auto iter = std::find(remaining.begin(), remaining.end(), member);
                         return iter == remaining.end();
                     });
        return out;
    }();

    std::unordered_map<int, int> index_from_id;
    for (int i = 0; i < static_cast<int>(members_.size()); i++) {
        index_from_id[members_.at(i)] = i;
    }

    const auto sum_over_marginalized = [&to_marginalize, &index_from_id,
                                        this](const Eigen::VectorXd &x) {
        const int n = to_marginalize.size();
        std::vector<double> terms;
        terms.reserve(1 << n);
        for (int num_present = 0; num_present <= n; num_present++) {
            // For each number of present beacons
            for (const auto &config : math::combinations(n, num_present)) {
                // We have a different way of that many beacons being present

                // Set the element for the current config
                Eigen::VectorXd curr_config = x;
                for (const int to_marginalize_idx : config) {
                    const int marginal_id = to_marginalize.at(to_marginalize_idx);
                    const int x_idx = index_from_id[marginal_id];
                    curr_config(x_idx) = 1;
                }

                // Evaluate the log probability
                terms.push_back(curr_config.transpose() * precision_ * curr_config -
                                log_normalizer_);
            }
        }
        return logsumexp(terms);
    };

    const int n_remaining = remaining.size();
    std::vector<LogMarginal> out;
    out.reserve(1 << n_remaining);
    for (int num_present = 0; num_present <= n_remaining; num_present++) {
        for (const auto &config : math::combinations(n_remaining, num_present)) {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(members_.size());
            std::vector<int> present_beacons;
            for (const int remaining_idx : config) {
                const int present_id = remaining.at(remaining_idx);
                present_beacons.push_back(present_id);
                const int x_idx = index_from_id.at(present_id);
                x(x_idx) = 1;
            }

            out.emplace_back(LogMarginal{
                .present_beacons = std::move(present_beacons),
                .log_marginal = sum_over_marginalized(x),
            });
        }
    }

    return out;
}

BeaconPotential create_correlated_beacons(const BeaconClique &clique) {
    drake::solvers::MathematicalProgram program;
    const auto precision_matrix_vars = program.NewContinuousVariables(2, "prec");
    const auto &diag = precision_matrix_vars[0];
    const auto &off_diag = precision_matrix_vars[1];
    const auto log_norm_var = program.NewContinuousVariables(1, "offset");
    const auto &log_norm = log_norm_var[0];
    program.AddLinearEqualityConstraint(-log_norm, std::log(clique.p_no_beacons));
    program.AddConstraint(compute_total_log_prob(clique.members.size(), diag, off_diag, log_norm) ==
                          0.0);
    program.AddConstraint(compute_marginal_log_prob(clique.members.size(), diag, off_diag,
                                                    log_norm) == std::log(clique.p_beacon));
    program.AddCost(diag * diag + off_diag * off_diag + log_norm * log_norm);
    // Set an initial guess away from 0.0. I'm not sure why, but it seems like the gradient is NaN
    // when all parameters equal 0.0.
    program.SetInitialGuess(diag, 1.0);
    program.SetInitialGuess(off_diag, 1.0);
    program.SetInitialGuess(log_norm, 1.0);

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

    return BeaconPotential(precision, result.GetSolution(log_norm), clique.members);
}
}  // namespace robot::experimental::beacon_sim
