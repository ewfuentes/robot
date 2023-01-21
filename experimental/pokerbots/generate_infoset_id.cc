
#include "experimental/pokerbots/generate_infoset_id.hh"

#include <limits>
#include <random>
#include <variant>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/hand_evaluator.hh"

namespace robot::experimental::pokerbots {
domain::RobPoker::InfoSetId infoset_id_from_history(const domain::RobPokerHistory &history,
                                                    const proto::PerTurnBinCenters &bin_centers,
                                                    InOut<std::mt19937> gen) {
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
        public_cards, history.actions, betting_state, bin_centers, gen);
}

domain::RobPoker::InfoSetId infoset_id_from_information(
    const std::array<domain::StandardDeck::Card, 2> &private_cards,
    const std::vector<domain::StandardDeck::Card> &common_cards,
    const std::vector<domain::RobPokerAction> &actions, const domain::BettingState &betting_state,
    const proto::PerTurnBinCenters &bin_centers, InOut<std::mt19937> gen) {
    using Suits = domain::StandardDeck::Suits;
    // There can be up to 6 actions, we allocate 4 bits for each actions
    // bits 40 - 63: observed actions
    // bits 32 - 39: betting round
    // bits 0 - 31: bucket idx
    uint64_t out = 0;
    // We omit the blinds as actions
    const int num_actions_this_round =
        betting_state.to_bet->position - (betting_state.to_bet->round == 0 ? 2 : 0);
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
    const int betting_round = betting_state.to_bet->round < 3
                                  ? betting_state.to_bet->round
                                  : (betting_state.to_bet->is_final_betting_round ? 3 : 2);

    out = (out << 8) | betting_round;

    if (betting_round == 0) {
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
            lower_card.suit == Suits::HEARTS || lower_card.suit == Suits::DIAMONDS;
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
        constexpr std::optional<time::RobotTimestamp::duration> timeout = {};
        constexpr std::optional<int> hand_limit = 250;
        constexpr int max_additional_cards = 2;
        const StrengthPotentialResult result = evaluate_strength_potential(
            private_cards, common_cards, max_additional_cards, timeout, hand_limit, gen);

        const auto &turn_bin_centers = [&bin_centers, betting_round]() {
            if (betting_round == 1) {
                return bin_centers.flop_centers();
            } else if (betting_round == 2) {
                return bin_centers.turn_centers();
            } else {
                return bin_centers.river_centers();
            }
        }();

        const auto dist_to_center = [&result](const auto &bin_center) {
            const double d_strength = result.strength - bin_center.strength();
            const double d_neg_pot = result.negative_potential - bin_center.negative_potential();
            const double d_pos_pot = result.positive_potential - bin_center.positive_potential();
            return std::hypot(d_strength, d_neg_pot, d_pos_pot);
        };

        const auto min_bucket_iter =
            std::min_element(turn_bin_centers.begin(), turn_bin_centers.end(),
                             [&dist_to_center](const auto &a, const auto &b) {
                                 return dist_to_center(a) < dist_to_center(b);
                             });
        const int bucket_idx = std::distance(turn_bin_centers.begin(), min_bucket_iter);

        out = (out << 32) | bucket_idx;
    }
    return out;
}
}  // namespace robot::experimental::pokerbots
