
#pragma once

#include <string>

#include "common/argument_wrapper.hh"
#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"

namespace robot::experimental::pokerbots {

struct ExpectedStrengthResult {
    double strength;
    uint64_t num_evaluations;
};
struct StrengthPotentialResult {
    double strength;
    double strength_potential;
    double positive_potential;
    double negative_potential;
    uint64_t num_evaluations;
};

struct HandDistributionResult {
    std::vector<uint64_t> distribution;
    uint64_t num_board_rollouts;
};

ExpectedStrengthResult evaluate_expected_strength(const std::string &hand,
                                                  const std::string &opponent_hand_str,
                                                  const std::string &board_str,
                                                  const std::optional<double> timeout_s,
                                                  const std::optional<int> num_hands);

StrengthPotentialResult evaluate_strength_potential(const std::string &hand_str,
                                                    const std::string &board_str,
                                                    const std::optional<int> max_additional_cards,
                                                    const std::optional<double> timeout_s,
                                                    const std::optional<int> num_hands);

HandDistributionResult estimate_hand_distribution(const std::string &hand_str,
                                                  const std::string &board_str, const int num_bins,
                                                  const std::optional<int> num_board_rollouts,
                                                  const std::optional<int> max_additional_cards,
                                                  const std::optional<double> timeout_s);

StrengthPotentialResult evaluate_strength_potential(
    const std::array<domain::StandardDeck::Card, 2> &hand,
    const std::vector<domain::StandardDeck::Card> &board,
    const std::optional<int> max_additional_cards,
    const std::optional<time::RobotTimestamp::duration> timeout, const std::optional<int> num_hands,
    InOut<std::mt19937> gen);

StrengthPotentialResult evaluate_strength_potential(
    const domain::RobPokerHistory &history, const domain::RobPokerPlayer player,
    const std::optional<int> max_additional_cards,
    const std::optional<time::RobotTimestamp::duration> timeout, const std::optional<int> num_hands,
    InOut<std::mt19937> gen);

HandDistributionResult estimate_hand_distribution(
    const std::array<domain::StandardDeck::Card, 2> &hand,
    const std::vector<domain::StandardDeck::Card> &board, const int num_bins,
    const std::optional<int> num_board_rollouts, const std::optional<int> max_additional_cards,
    const std::optional<time::RobotTimestamp::duration> timeout, InOut<std::mt19937> gen);

}  // namespace robot::experimental::pokerbots
