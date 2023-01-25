

#include <pstl/glue_execution_defs.h>

#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "experimental/pokerbots/hand_evaluator.hh"

namespace robot::experimental::pokerbots {

namespace {
std::vector<domain::StandardDeck::Card> cards_from_string(const std::string &cards_str) {
    const std::string ranks = "23456789TJQKA";
    const std::string suits = "shcd";
    std::vector<domain::StandardDeck::Card> out;
    for (int i = 0; i < static_cast<int>(cards_str.size()); i += 2) {
        out.push_back({
            static_cast<domain::StandardRanks>(ranks.find(cards_str[i])),
            static_cast<domain::StandardSuits>(suits.find(cards_str[i + 1])),
        });
    }
    return out;
}
}  // namespace

std::vector<StrengthPotentialResult> evaluate_mass_hands(const std::vector<std::string> &hands,
                                                         const int max_additional_cards,
                                                         const int hands_per_eval) {
    using HoleCards = std::array<domain::StandardDeck::Card, 2>;
    using BoardCards = std::vector<domain::StandardDeck::Card>;
    using EvalInput = std::tuple<int, HoleCards, BoardCards>;

    std::vector<EvalInput> inputs;
    inputs.reserve(hands.size());
    for (int i = 0; i < static_cast<int>(hands.size()); i++) {
        auto cards = cards_from_string(hands[i]);
        inputs.emplace_back(
            EvalInput(i, {cards[0], cards[1]}, BoardCards(cards.begin() + 2, cards.end())));
    }

    constexpr int NUM_CHUNKS = 100;
    const int chunk_size = hands.size() / NUM_CHUNKS;

    std::vector<int> chunk_starts;
    for (int i = 0; i < static_cast<int>(hands.size()); i += chunk_size) {
        chunk_starts.push_back(i);
    }

    std::vector<StrengthPotentialResult> outputs(hands.size());
    std::cout << "Num Chunks: " << chunk_starts.size() << std::endl;
    std::for_each(
        std::execution::par, chunk_starts.begin(), chunk_starts.end(),
        [&inputs, &outputs, chunk_size, max_additional_cards,
         hands_per_eval](const int chunk_start) {
            const auto start = time::current_robot_time();
            std::cout << "Starting chunk at " << chunk_start << std::endl;
            for (int i = chunk_start;
                 i < chunk_start + chunk_size && i < static_cast<int>(inputs.size()); i++) {
                const auto &[idx, hole_cards, board_cards] = inputs[i];
                std::mt19937 gen(idx);
                outputs[idx] =
                    evaluate_strength_potential(hole_cards, board_cards, max_additional_cards, {},
                                                hands_per_eval, make_in_out(gen));
            }
            const auto dt = time::current_robot_time() - start;
            std::cout << "Chunk at " << chunk_start << " finished in: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(dt).count()
                      << std::endl;
        });
    return outputs;
}

std::vector<HandDistributionResult> mass_estimate_hand_distribution(
    const std::vector<std::string> &hands, const int max_additional_cards,
    const int num_board_rollouts, const int num_bins) {
    using HoleCards = std::array<domain::StandardDeck::Card, 2>;
    using BoardCards = std::vector<domain::StandardDeck::Card>;
    using EvalInput = std::tuple<int, HoleCards, BoardCards>;

    std::vector<EvalInput> inputs;
    inputs.reserve(hands.size());
    for (int i = 0; i < static_cast<int>(hands.size()); i++) {
        auto cards = cards_from_string(hands[i]);
        inputs.emplace_back(
            EvalInput(i, {cards[0], cards[1]}, BoardCards(cards.begin() + 2, cards.end())));
    }

    std::vector<HandDistributionResult> out(hands.size());
    std::for_each(std::execution::par, inputs.begin(), inputs.end(), [&](const EvalInput &input) {
        const auto &[idx, hand, board] = input;
        std::mt19937 gen(idx);
        out[idx] = estimate_hand_distribution(hand, board, num_bins, num_board_rollouts,
                                              max_additional_cards, {}, make_in_out(gen));
    });
    return out;
}
}  // namespace robot::experimental::pokerbots
