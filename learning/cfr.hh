

#pragma once

#include <functional>
#include <iterator>
#include <numeric>
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
    std::optional<Strategy<T>> operator()(const typename T::History &history) const;
    std::optional<Strategy<T>> operator()(const typename T::InfoSetId &id) const ;

    std::unordered_map<typename T::InfoSetId, Strategy<T>> strategy_from_infoset_id;
    std::function<typename T::InfoSetId(typename T::History)> infoset_id_from_hist;
};

template <typename T>
struct MinRegretTrainConfig {
    int num_iterations;
    std::function<typename T::InfoSetId(const typename T::History &)> infoset_id_from_hist;
    int seed;
};

template <typename T>
MinRegretStrategy<T> train_min_regret_strategy(const MinRegretTrainConfig<T> &config);

// Implementations
template <typename T>
std::optional<Strategy<T>> MinRegretStrategy<T>::operator()(const typename T::History &history) const {
    return (*this)(infoset_id_from_hist(history));
}

template <typename T>
std::optional<Strategy<T>> MinRegretStrategy<T>::operator()(const typename T::InfoSetId &id) const {
  const auto iter = strategy_from_infoset_id.find(id);
  if (iter == strategy_from_infoset_id.end()) {
    return std::nullopt;
  }
  return iter->second;
}

template <typename T>
struct InfoSetCounts {
    IndexedArray<double, typename T::Actions> regret_sum = {0};
    IndexedArray<double, typename T::Actions> strategy_sum = {0};
    int iter_count = 0;
};

template <typename T>
Strategy<T> strategy_from_counts(const InfoSetCounts<T> &counts) {
    // Compute the normalization factor
    const double partition =
        std::accumulate(counts.regret_sum.begin(), counts.regret_sum.end(), 0.0,
                        [](const double positive_regret_sum, const auto &action_and_regret) {
                            return positive_regret_sum + std::max(action_and_regret.second, 0.0);
                        });

    // Select action in proportion to positive regrets
    Strategy<T> out;
    for (const auto &[action, regret] : counts.regret_sum) {
        if (partition == 0.0) {
            out[action] = 1.0 / out.size();
        } else {
            out[action] = std::max(regret, 0.0) / partition;
        }
    }
    return out;
}

template <typename T>
std::unordered_map<typename T::InfoSetId, Strategy<T>> compute_average_strategy(
    const std::unordered_map<typename T::InfoSetId, InfoSetCounts<T>> &counts_from_infoset_id) {
    std::unordered_map<typename T::InfoSetId, Strategy<T>> out;
    std::transform(counts_from_infoset_id.begin(), counts_from_infoset_id.end(),
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
    return out;
}

template <typename T>
MinRegretStrategy<T> train_min_regret_strategy(const MinRegretTrainConfig<T> &config) {
    std::mt19937 gen(config.seed);
    std::unordered_map<typename T::InfoSetId, InfoSetCounts<T>> counts_from_infoset_id;

    std::vector<typename T::Players> non_chance_players;
    for (const auto &[player, name] : wise_enum::range<typename T::Players>) {
        constexpr bool has_chance = requires { T::Players::CHANCE; };
        if constexpr (has_chance) {
            if (player == T::Players::CHANCE) {
                continue;
            }
        }
        non_chance_players.push_back(player);
    }

    for (int iter = 0; iter < config.num_iterations; iter++) {
        for (auto player : non_chance_players) {
            compute_counterfactual_regret({}, player, iter, {1.0}, config.infoset_id_from_hist,
                                          make_in_out(gen), make_in_out(counts_from_infoset_id));
        }
    }

    // Output the average strategy
    return {.strategy_from_infoset_id = compute_average_strategy(counts_from_infoset_id),
            .infoset_id_from_hist = config.infoset_id_from_hist};
}

template <typename T>
double compute_counterfactual_regret(
    const typename T::History &history, const typename T::Players to_update, const int iteration,
    const IndexedArray<double, typename T::Players> &action_probabilities,
    const std::function<typename T::InfoSetId(const typename T::History &)>
        &infoset_id_from_history,
    InOut<std::mt19937> gen,
    InOut<std::unordered_map<typename T::InfoSetId, InfoSetCounts<T>>> counts_from_infoset_id) {
    // If this is a terminal state, return terminal_value
    const auto maybe_current_player = up_next(history);
    if (!maybe_current_player.has_value()) {
        return terminal_value(history, to_update).value();
    }

    const auto &current_player = maybe_current_player.value();
    // If this is a chance state, sample it and recurse
    constexpr bool has_chance = requires { T::Players::CHANCE; };
    if constexpr (has_chance) {
        if (current_player == T::Players::CHANCE) {
            const auto chance_result = play(history, gen);
            auto new_action_probabilities = action_probabilities;
            new_action_probabilities[T::Players::CHANCE] *= chance_result.probability;
            return compute_counterfactual_regret<T>(
                chance_result.history, to_update, iteration, new_action_probabilities,
                infoset_id_from_history, gen, counts_from_infoset_id);
        }
    }

    // Compute the value of the current node
    double node_value = 0;
    IndexedArray<double, typename T::Actions> action_values;
    // Note that this isn't const because we may update it later
    // We want to construct a new InfoSetCounts if it doesn't already exist, so we use [] instead of
    // at()
    InfoSetCounts<T> &counts = (*counts_from_infoset_id)[infoset_id_from_history(history)];
    Strategy<T> strategy = strategy_from_counts(counts);

    auto new_action_probabilities = action_probabilities;
    for (const auto &[action, probability] : strategy) {
        new_action_probabilities[current_player] =
            action_probabilities[current_player] * probability;
        action_values[action] = compute_counterfactual_regret(
            play(history, action), to_update, iteration, new_action_probabilities,
            infoset_id_from_history, gen, counts_from_infoset_id);
        node_value += probability * action_values[action];
    }

    // Update the counts for the current player if appropriate
    if (current_player == to_update) {
        const double opponent_probability = std::accumulate(
            action_probabilities.begin(), action_probabilities.end(), 1.0,
            [current_player](const double probability, const auto &player_and_probability) {
                const auto &[player, player_probability] = player_and_probability;
                return probability * (player == current_player ? 1.0 : player_probability);
            });
        for (const auto &[action, action_value] : action_values) {
            counts.regret_sum[action] += opponent_probability * (action_value - node_value);
            counts.strategy_sum[action] += action_probabilities[current_player] * strategy[action];
        }
    }
    return node_value;
}
}  // namespace robot::learning
