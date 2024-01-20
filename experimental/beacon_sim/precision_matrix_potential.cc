
#include "experimental/beacon_sim/precision_matrix_potential.hh"

#include <algorithm>
#include <iterator>
#include <numeric>

#include "common/check.hh"
#include "common/math/combinations.hh"

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

}  // namespace

std::vector<int> get_members(const PrecisionMatrixPotential &pot) { return pot.members; }

double compute_log_prob(const PrecisionMatrixPotential &pot,
                        const std::unordered_map<int, bool> &assignment,
                        const bool allow_partial_assignment) {
    const std::vector<int> sorted_members = sorted_vector(pot.members);
    const std::vector<int> keys = sorted_keys(assignment);
    std::vector<int> missing_keys;
    std::set_difference(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                        std::back_inserter(missing_keys));

    std::vector<int> keys_to_keep;
    std::set_intersection(sorted_members.begin(), sorted_members.end(), keys.begin(), keys.end(),
                          std::back_inserter(keys_to_keep));

    CHECK(allow_partial_assignment || missing_keys.empty(),
          "partial assignment specified when not enabled", assignment, missing_keys, pot.members);

    const std::vector<int> to_marginalize = missing_keys;

    std::unordered_map<int, int> index_from_id;
    for (int i = 0; i < static_cast<int>(pot.members.size()); i++) {
        index_from_id[pot.members.at(i)] = i;
    }

    const auto sum_over_marginalized = [&to_marginalize, &index_from_id,
                                        &pot](const Eigen::VectorXd &x) {
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
                terms.push_back(curr_config.transpose() * pot.precision * curr_config -
                                pot.log_normalizer);
            }
        }
        return logsumexp(terms);
    };

    Eigen::VectorXd config = Eigen::VectorXd::Zero(pot.members.size());
    for (const auto beacon_id : keys_to_keep) {
        config(index_from_id.at(beacon_id)) = assignment.at(beacon_id);
    }

    return sum_over_marginalized(config);
}

std::vector<LogMarginal> compute_log_marginals(const PrecisionMatrixPotential &pot,
                                               const std::vector<int> &all_remaining) {
    const auto sorted_all_remaining = sorted_vector(all_remaining);
    const auto sorted_pot_members = sorted_vector(pot.members);
    std::vector<int> remaining;
    std::set_intersection(sorted_all_remaining.begin(), sorted_all_remaining.end(),
                          sorted_pot_members.begin(), sorted_pot_members.end(),
                          std::back_inserter(remaining));

    // Find the members that we need to marginalize over
    const std::vector<int> to_marginalize = [&]() {
        std::vector<int> out;
        std::copy_if(pot.members.begin(), pot.members.end(), std::back_inserter(out),
                     [&](const int &member) {
                         const auto iter = std::find(remaining.begin(), remaining.end(), member);
                         return iter == remaining.end();
                     });
        return out;
    }();

    std::unordered_map<int, int> index_from_id;
    for (int i = 0; i < static_cast<int>(pot.members.size()); i++) {
        index_from_id[pot.members.at(i)] = i;
    }

    const auto sum_over_marginalized = [&to_marginalize, &index_from_id,
                                        &pot](const Eigen::VectorXd &x) {
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
                terms.push_back(curr_config.transpose() * pot.precision * curr_config -
                                pot.log_normalizer);
            }
        }
        return logsumexp(terms);
    };

    const int n_remaining = remaining.size();
    std::vector<LogMarginal> out;
    out.reserve(1 << n_remaining);
    for (int num_present = 0; num_present <= n_remaining; num_present++) {
        for (const auto &config : math::combinations(n_remaining, num_present)) {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(pot.members.size());
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

std::vector<int> generate_sample(const PrecisionMatrixPotential &pot, InOut<std::mt19937> gen) {
    double remaining_prob = std::uniform_real_distribution<>()(*gen);

    std::unordered_map<int, bool> all_gone;
    for (const int beacon_id : pot.members) {
        all_gone[beacon_id] = false;
    }
    std::vector<int> out;
    for (int num_present = 0; num_present <= static_cast<int>(pot.members.size()); num_present++) {
        for (const auto &idxs : math::combinations(pot.members.size(), num_present)) {
            out.clear();
            std::unordered_map config = all_gone;
            for (const int idx : idxs) {
                out.push_back(pot.members.at(idx));
                config.at(pot.members.at(idx)) = true;
            }
            constexpr bool DONT_ALLOW_PARTIAL_ASSIGNMENT = false;
            const double log_prob = compute_log_prob(pot, config, DONT_ALLOW_PARTIAL_ASSIGNMENT);

            remaining_prob -= std::exp(log_prob);
            if (remaining_prob < 0) {
                return out;
            }
        }
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim
