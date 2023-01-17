
#include "experimental/pokerbots/generate_infoset_id.hh"

#include <variant>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/hand_evaluator.hh"

namespace robot::experimental::pokerbots {
uint64_t low_counts = 0;
uint64_t high_counts = 0;
domain::RobPoker::InfoSetId infoset_id_from_history(const domain::RobPokerHistory &history) {
    const auto player = up_next(history).value();
    const auto betting_state = compute_betting_state(history);
    std::vector<domain::StandardDeck::Card> public_cards;
    public_cards.reserve(history.common_cards.size());
    for (const auto &card : history.common_cards) {
        if (card.has_value() && card.is_visible_to(player)) {
            public_cards.push_back(card.value());
        }
    }

    return infoset_id_from_information(
        {history.hole_cards[player][0].value(), history.hole_cards[player][1].value()},
        public_cards, history.actions, betting_state);
}

domain::RobPoker::InfoSetId infoset_id_from_information(
    const std::array<domain::StandardDeck::Card, 2> &private_cards,
    const std::vector<domain::StandardDeck::Card> &common_cards,
    const std::vector<domain::RobPokerAction> &actions, const domain::BettingState &betting_state) {
    using Suits = domain::StandardDeck::Suits;
    // There can be up to 6 actions, we allocate 4 bits for each actions
    // bits 40 - 63: observed actions
    // bits 32 - 39: betting round
    // bits 0 - 31: bucket idx
    uint64_t out = 0;
    // We omit the blinds as actions
    const int num_actions_this_round = betting_state.position - (betting_state.round == 0 ? 2 : 0);
    // If the action is going to fit in 4 bits, there need to be fewer than 14 options
    static_assert(std::variant_size_v<domain::RobPokerAction> < 15);
    for (int idx_offset = 0; idx_offset < num_actions_this_round; idx_offset++) {
        const int idx = actions.size() - 1 - idx_offset;
        out = (out << 4) | (actions.at(idx).index() + 1);
    }

    // Betting Rounds:
    //  0 - Preflop
    //  1 - Postflop/pre turn
    //  2 - Preriver/ prerun
    //  3 - Final betting round
    const int betting_round = betting_state.round < 3
                                  ? betting_state.round
                                  : (betting_state.is_final_betting_round ? 3 : 2);

    out = (out << 8) | betting_round;

    if (betting_state.round == 0) {
        low_counts++;
        // Map the hole cards into a bucket
        // bits 16 - 31: rank bit mask
        out = (out << 16);
        for (const auto card : private_cards) {
            out |= 1 << static_cast<int>(card.rank);
        }
        // bits 0-15: suits enum
        //  0 - suited black
        //  1 - suited red
        //  2 - offsuit both black
        //  3 - offsuit both red
        //  4 - offsuit higher black
        //  5 - offsuit higher red
        out = (out << 16);
        const auto &higher_card =
            private_cards[0].rank > private_cards[1].rank ? private_cards[0] : private_cards[1];
        const auto &lower_card =
            private_cards[0].rank > private_cards[1].rank ? private_cards[1] : private_cards[0];
        const bool is_higher_red =
            higher_card.suit == Suits::HEARTS || higher_card.suit == Suits::DIAMONDS;
        const bool is_lower_red =
            lower_card.suit == Suits::HEARTS || higher_card.suit == Suits::DIAMONDS;
        const bool is_suit_equal = higher_card.suit == lower_card.suit;

        if (is_suit_equal) {
            // Suited black or red
            out |= (is_higher_red ? 1 : 0);
        } else if (is_higher_red == is_lower_red) {
            // off suit, both black or both red
            out |= (is_higher_red ? 3 : 2);
        } else {
            // off suit of different colors
            out |= (is_higher_red ? 5 : 4);
        }
    } else {
        high_counts++;
        constexpr std::optional<time::RobotTimestamp::duration> timeout = {};
        constexpr std::optional<int> hand_limit = 1000;
        const StrengthPotentialResult result =
            evaluate_strength_potential(private_cards, common_cards, timeout, hand_limit);
        // Bin the hand by expected hand strength, positve and negative potential
        // Hand strength bins:
        //  0 - [0.0, 0.2)
        //  1 - [0.2, 0.4)
        //  2 - [0.4, 0.6)
        //  3 - [0.6, 0.8)
        //  4 - [0.8, 1.0]
        int hand_strength_bin = 0;
        if (result.strength < 0.2) {
            hand_strength_bin = 0;
        } else if (result.strength < 0.4) {
            hand_strength_bin = 1;
        } else if (result.strength < 0.6) {
            hand_strength_bin = 2;
        } else if (result.strength < 0.8) {
            hand_strength_bin = 3;
        } else {
            hand_strength_bin = 4;
        }

        // Negative/Postive potential bins:
        // 0 - [0.0, 0.1)
        // 1 - [0.1, 0.2)
        // 2 - [0.2, 1.0]
        int negative_potential_bin = 0;
        if (result.negative_potential < 0.1) {
            negative_potential_bin = 0;
        } else if (result.negative_potential < 0.2) {
            negative_potential_bin = 1;
        } else {
            negative_potential_bin = 2;
        }

        int positive_potential_bin = 0;
        if (result.positive_potential < 0.1) {
            positive_potential_bin = 0;
        } else if (result.positive_potential < 0.2) {
            positive_potential_bin = 1;
        } else {
            positive_potential_bin = 2;
        }

        out |= (hand_strength_bin << 8) | (negative_potential_bin << 4) | (positive_potential_bin);
        // 5 * 3 * 3 bins = 45 bins
    }
    return out;
}
}  // namespace robot::experimental::pokerbots
