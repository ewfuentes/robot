

#pragma once

#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <unordered_map>

#include "common/argument_wrapper.hh"
#include "common/indexed_array.hh"
#include "wise_enum.h"

namespace robot::learning {

// Declarations
template <typename T>
using Strategy = IndexedArray<double, typename T::Actions>;

template <typename T>
std::string to_string(const Strategy<T> &strategy) {
    std::stringstream out;
    out << "[";
    for (const auto &[action, prob] : strategy) {
        out << wise_enum::to_string(action) << ": " << prob << " ";
    }
    out.seekp(-1, std::ios_base::end);
    out << "]";
    return out.str();
}

template <typename T>
typename T::Actions sample_strategy(const Strategy<T> &strategy, InOut<std::mt19937> gen) {
    double rand_value = std::uniform_real_distribution<>()(*gen);
    for (const auto &[action, prob] : strategy) {
        rand_value -= prob;
        if (rand_value < 0.0) {
            return action;
        }
    }
    return wise_enum::range<typename T::Actions>.begin()->value;
}

template <typename T>
struct MinRegretStrategy {
    std::optional<Strategy<T>> get_strategy(const typename T::History &history);
    Strategy<T> operator()(const typename T::History &history);

    std::unordered_map<typename T::InfoSetId, Strategy<T>> strategy_from_info_set_id;
};

template <typename T>
struct MinRegretTrainConfig {
    int num_iterations;
    IndexedArray<std::function<Strategy<T>(const typename T::History &)>, typename T::Players>
        fixed_strategies;
    std::function<typename T::InfoSetId(const typename T::History &)> info_set_id_from_hist;
    int seed;
};

template <typename T>
MinRegretStrategy<T> train_min_regret_strategy(const MinRegretTrainConfig<T> &config);

// Implementations
template <typename T>
std::optional<Strategy<T>> MinRegretStrategy<T>::get_strategy(const typename T::History &history) {
    (void)history;
    return std::nullopt;
}

template <typename T>
MinRegretStrategy<T> train_min_regret_strategy(const MinRegretTrainConfig<T> &config) {
    std::mt19937 gen(config.seed);
    struct InfoSetCounts {
        IndexedArray<double, typename T::Actions> regret_sum = {0};
        IndexedArray<double, typename T::Actions> strategy_sum = {0};
        int iter_count = 0;
    };
    std::unordered_map<typename T::InfoSetId, InfoSetCounts> regrets_from_infoset;

    const auto get_min_regret_strategy =
        [&regrets_from_infoset,
         &info_set_id_from_hist = config.info_set_id_from_hist](const auto &hist) -> Strategy<T> {
        const auto info_set_id = info_set_id_from_hist(hist);
        const auto &info_set_counts = regrets_from_infoset[info_set_id];
        // Compute the normalization factor
        const double partition = std::accumulate(
            info_set_counts.regret_sum.begin(), info_set_counts.regret_sum.end(), 0.0,
            [](const double positive_regret_sum, const auto &action_and_regret) {
                return positive_regret_sum + std::max(action_and_regret.second, 0.0);
            });

        // Select action in proportion to positive regrets
        Strategy<T> out;
        for (const auto &[action, regret] : info_set_counts.regret_sum) {
            if (partition == 0.0) {
                out[action] = 1.0 / out.size();
            } else {
                out[action] = std::max(regret, 0.0) / partition;
            }
        }
        return out;
    };

    for (int iter = 0; iter < config.num_iterations; iter++) {
        auto history = typename T::History();
        // Sample Actions for each player
        while (true) {
            const auto maybe_next_player = up_next(history);
            if (!maybe_next_player.has_value()) {
                break;
            }

            const auto maybe_fixed_strategy = config.fixed_strategies[maybe_next_player.value()];
            const auto strategy = maybe_fixed_strategy ? maybe_fixed_strategy(history)
                                                       : get_min_regret_strategy(history);

            const auto action = sample_strategy<T>(strategy, make_in_out(gen));
            history = play(history, action);
        }

        // Update the regrets for each player
        for (const auto &[player, _] : wise_enum::range<typename T::Players>) {
            if (config.fixed_strategies[player]) {
                continue;
            }

            const auto info_set_id = config.info_set_id_from_hist(history);
            auto &counts = regrets_from_infoset[info_set_id];
            // Update the regret sum
            for (const auto &[new_action, _] : wise_enum::range<typename T::Actions>) {
                counts.regret_sum[new_action] +=
                    compute_counterfactual_regret(history, player, new_action);
            }
            for (const auto &[action, regret] : counts.regret_sum) {
                counts.strategy_sum[action] += std::max(0.0, regret);
            }
            counts.iter_count++;
        }
    }

    // Output the average strategy
    std::unordered_map<typename T::InfoSetId, Strategy<T>> out;
    std::transform(regrets_from_infoset.begin(), regrets_from_infoset.end(),
                   std::inserter(out, out.begin()),
                   [](const auto &id_and_counts) -> std::pair<typename T::InfoSetId, Strategy<T>> {
                       const auto &[id, counts] = id_and_counts;
                       const double partition = std::accumulate(
                           counts.strategy_sum.begin(), counts.strategy_sum.end(), 0.0,
                           [](const double positive_regret_sum, const auto &action_and_regret) {
                               return positive_regret_sum + std::max(action_and_regret.second, 0.0);
                           });
                       Strategy<T> strategy;
                       for (const auto &&[action, prob] : strategy) {
                           prob = counts.strategy_sum[action] / partition;
                       }
                       return {id, strategy};
                   });

    return {.strategy_from_info_set_id = out};
}
}  // namespace robot::learning
