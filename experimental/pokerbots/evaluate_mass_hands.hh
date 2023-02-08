
#pragma once

#include <vector>

#include "experimental/pokerbots/hand_evaluator.hh"

namespace robot::experimental::pokerbots {
std::vector<StrengthPotentialResult> evaluate_mass_hands(const std::vector<std::string> &hands,
                                                         const int max_additional_cards,
                                                         const int hands_per_eval);

std::vector<HandDistributionResult> mass_estimate_hand_distribution(
    const std::vector<std::string> &hands, const int max_additional_cards,
    const int num_board_rollouts, const int num_bins);
}  // namespace robot::experimental::pokerbots
