
#pragma once

#include <string>

#include "common/argument_wrapper.hh"
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

ExpectedStrengthResult evaluate_expected_strength(const std::string &hand,
                                                  const std::string &opponent_hand_str,
                                                  const std::string &board_str,
                                                  const double timeout_s);

StrengthPotentialResult evaluate_strength_potential(const std::string &hand_str,
                                                    const std::string &board_str,
                                                    const double timeout_s);

StrengthPotentialResult evaluate_strength_potential(
    const std::array<domain::StandardDeck::Card, 2> &hand,
    const std::vector<domain::StandardDeck::Card> &board, const double timeout_s);

StrengthPotentialResult evaluate_strength_potential(const domain::RobPokerHistory &history,
                                                    const domain::RobPokerPlayer player,
                                                    InOut<std::mt19937> gen);
}  // namespace robot::experimental::pokerbots
