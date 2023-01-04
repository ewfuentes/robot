

#pragma once

#include <functional>
#include <optional>
#include <random>
#include <unordered_map>

#include "common/argument_wrapper.hh"
#include "common/indexed_array.hh"
#include "wise_enum.h"

namespace robot::learning {

// Declarations
template <typename T>
using Strategy = IndexedArray<double, typename T::Actions>;

template <typename T>
class MinRegretStrategy {
   public:
    std::optional<Strategy<T>> get_strategy(const typename T::History &history);
    Strategy<T> operator()(const typename T::History &history);

   private:
    std::unordered_map<typename T::InfoSetId, Strategy<T>> strategy_by_info_set_id;
};

template <typename T>
struct MinRegretTrainConfig {
    int num_iterations;
    std::unordered_map<typename T::Players, std::function<Strategy<T>(const typename T::History &)>>
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
    struct InfoSetCounts {
        IndexedArray<double, typename T::Actions> regret_sum = {0};
        IndexedArray<double, typename T::Actions> strategy_sum = {0};
    };
    std::unordered_map<typename T::InfoSetId, InfoSetCounts> regrets_from_infoset;

    const auto get_min_regret_strategy =
        [&regrets_from_infoset,
         info_set_id_from_hist = &config.info_set_id_from_hist](const auto &hist) -> Strategy<T> {
        const auto info_set_id = info_set_id_from_hist(hist);
        const auto &info_set_counts = regrets_from_infoset[info_set_id];

        return {};
    };
    (void)get_min_regret_strategy;

    for (int iter = 0; iter < config.num_iterations; iter++) {
        const auto history = typename T::History();
        // Sample Actions for each player

        // Update the regrets for each player
    }
    return {};
}
}  // namespace robot::learning
