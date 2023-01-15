
#pragma once

#include <chrono>
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
struct InfoSetCounts {
    IndexedArray<double, typename T::Actions> regret_sum = {0};
    IndexedArray<double, typename T::Actions> strategy_sum = {0};
    int iter_count = 0;
};

template <typename T>
typename T::Actions sample_strategy(const Strategy<T> &strategy, InOut<std::mt19937> gen) {
    double rand_value = std::uniform_real_distribution<>()(*gen);
    for (const auto &[action, prob] : strategy) {
        rand_value -= prob;
        if (rand_value < 0.0) {
            return action;
        }
    }
    const auto &[value, _] = *Range<typename T::Actions>::value.begin();
    return value;
}

template <typename T>
struct MinRegretStrategy {
    std::optional<Strategy<T>> operator()(const typename T::History &history) const;
    std::optional<Strategy<T>> operator()(const typename T::InfoSetId &id) const;

    std::unordered_map<typename T::InfoSetId, Strategy<T>> strategy_from_infoset_id;
    std::function<typename T::InfoSetId(typename T::History)> infoset_id_from_hist;
};

WISE_ENUM_CLASS(SampleStrategy, CHANCE_SAMPLING, EXTERNAL_SAMPLING)

template <typename T>
struct MinRegretTrainConfig {
    int num_iterations;
    std::function<typename T::InfoSetId(const typename T::History &)> infoset_id_from_hist;
    std::function<std::vector<typename T::Actions>(const typename T::History &)> action_generator;
    int seed;
    SampleStrategy sample_strategy;
    std::function<bool(const int,
                       const std::unordered_map<typename T::InfoSetId, InfoSetCounts<T>>)>
    iteration_callback = [](const auto &, const auto &){return true;};
};

template <typename T>
MinRegretStrategy<T> train_min_regret_strategy(const MinRegretTrainConfig<T> &config);

// Implementations
template <typename T>
std::optional<Strategy<T>> MinRegretStrategy<T>::operator()(
    const typename T::History &history) const {
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
    bool run = true;
    for (int iter = 0; iter < config.num_iterations && run; iter++) {
        if (config.iteration_callback) {
            run = config.iteration_callback(iter, counts_from_infoset_id);
        }
        for (auto player : non_chance_players) {
            if (config.sample_strategy == SampleStrategy::CHANCE_SAMPLING) {
                compute_chance_sampled_counterfactual_regret(
                    {}, player, iter, {1.0}, config.infoset_id_from_hist, config.action_generator,
                    make_in_out(gen), make_in_out(counts_from_infoset_id));
            } else {
                compute_external_sampled_counterfactual_regret(
                    {}, player, iter, config.infoset_id_from_hist, config.action_generator,
                    make_in_out(gen), make_in_out(counts_from_infoset_id));
            }
        }
    }

    // Output the average strategy
    return {.strategy_from_infoset_id = compute_average_strategy(counts_from_infoset_id),
            .infoset_id_from_hist = config.infoset_id_from_hist};
}

template <typename T>
double compute_external_sampled_counterfactual_regret(
    const typename T::History &history, const typename T::Players to_update, const int iteration,
    const std::function<typename T::InfoSetId(const typename T::History &)>
        &infoset_id_from_history,
    const std::function<std::vector<typename T::Actions>(const typename T::History &)>
        &action_generator,
    InOut<std::mt19937> gen,
    InOut<std::unordered_map<typename T::InfoSetId, InfoSetCounts<T>>> counts_from_infoset_id) {
    const auto maybe_current_player = up_next(history);
    // If this is a leaf node, return the value
    if (!maybe_current_player.has_value()) {
        return terminal_value(history, to_update).value();
    }

    const auto &current_player = maybe_current_player.value();

    // If a chance player is up next, sample it and get its value
    constexpr bool has_chance = requires { T::Players::CHANCE; };
    if constexpr (has_chance) {
        if (current_player == T::Players::CHANCE) {
            const auto chance_result = play(history, gen);
            return compute_external_sampled_counterfactual_regret<T>(
                chance_result.history, to_update, iteration, infoset_id_from_history,
                action_generator, gen, counts_from_infoset_id);
        }
    }
    // Get the current strategy for this infoset. We have a non const reference since we
    // may update them further down.
    InfoSetCounts<T> &counts = (*counts_from_infoset_id)[infoset_id_from_history(history)];
    counts.iter_count++;
    const Strategy<T> maybe_invalid_strategy = strategy_from_counts(counts);
    Strategy<T> valid_strategy = {0.0};
    double normalizer = 0;
    const auto valid_actions = action_generator(history);
    {
        for (const auto &action : valid_actions) {
            normalizer += maybe_invalid_strategy[action];
            valid_strategy[action] = maybe_invalid_strategy[action];
        }
        for (const auto &action : valid_actions) {
            valid_strategy[action] = valid_strategy[action] / normalizer;
        }
    }

    if (current_player == to_update) {
        double node_value = 0.0;
        IndexedArray<double, typename T::Actions> action_values = {0.0};

        // Compute the node and action values
        for (const auto &action : valid_actions) {
            action_values[action] = compute_external_sampled_counterfactual_regret(
                play(history, action), to_update, iteration, infoset_id_from_history,
                action_generator, gen, counts_from_infoset_id);
            node_value += valid_strategy[action] * action_values[action];
        }

        // Update the regrets
        for (const auto &action : valid_actions) {
            const double counterfactual_regret = action_values[action] - node_value;
            counts.regret_sum[action] += counterfactual_regret;
        }
        return node_value;

    } else {
        // This is some player other than the one we are updating regrets for
        // Need to assign invalid strategies 0 probability an renormalize
        const auto action = sample_strategy<T>(valid_strategy, gen);
        const double value = compute_external_sampled_counterfactual_regret(
            play(history, action), to_update, iteration, infoset_id_from_history, action_generator,
            gen, counts_from_infoset_id);

        for (const auto &[action, probability] : valid_strategy) {
            counts.strategy_sum[action] += probability;
        }
        return value;
    }
}

template <typename T>
double compute_chance_sampled_counterfactual_regret(
    const typename T::History &history, const typename T::Players to_update, const int iteration,
    const IndexedArray<double, typename T::Players> &action_probabilities,
    const std::function<typename T::InfoSetId(const typename T::History &)>
        &infoset_id_from_history,
    const std::function<std::vector<typename T::Actions>(const typename T::History &)>
        &action_generator,
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
            return compute_chance_sampled_counterfactual_regret<T>(
                chance_result.history, to_update, iteration, action_probabilities,
                infoset_id_from_history, action_generator, gen, counts_from_infoset_id);
        }
    }

    const double opponent_probability = std::accumulate(
        action_probabilities.begin(), action_probabilities.end(), 1.0,
        [current_player](const double probability, const auto &player_and_probability) {
            const auto &[player, player_probability] = player_and_probability;
            return probability * (player == current_player ? 1.0 : player_probability);
        });
    if (opponent_probability == 0.0 && action_probabilities[current_player] == 0.0) {
        return 0.0;
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
    for (const auto &action : action_generator(history)) {
        const auto probability = strategy[action];
        new_action_probabilities[current_player] =
            action_probabilities[current_player] * probability;

        action_values[action] = compute_chance_sampled_counterfactual_regret(
            play(history, action), to_update, iteration, new_action_probabilities,
            infoset_id_from_history, action_generator, gen, counts_from_infoset_id);

        node_value += probability * action_values[action];
    }

    // Update the counts for the current player if appropriate
    if (current_player == to_update) {
        for (const auto &[action, action_value] : action_values) {
            counts.regret_sum[action] += opponent_probability * (action_value - node_value);
            counts.strategy_sum[action] += action_probabilities[current_player] * strategy[action];
        }
    }
    return node_value;
}
}  // namespace robot::learning
