
#include "domain/rob_poker.hh"

#include <ios>
#include <iostream>
#include <limits>
#include <sstream>

#include "domain/deck.hh"
#include "domain/fog.hh"
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"

namespace robot::domain {
namespace {
struct BettingState {
    bool is_game_over;
    bool showdown_required;
    int round;
    int position;
};

BettingState compute_betting_state(const RobPokerHistory &history) {
    using A = RobPokerAction;
    BettingState state = {
        .is_game_over = false, .showdown_required = false, .round = 0, .position = 0};
    if (history.actions.empty()) {
        return state;
    }
    for (int i = 0; i < static_cast<int>(history.actions.size()); i++) {
        const auto curr_action = history.actions.at(i);
        bool did_advance_to_new_betting_round = false;
        if (curr_action == A::FOLD) {
            state.is_game_over = true;
            return state;
        } else if (i > 0) {
            const auto prev_action = history.actions.at(i - 1);
            const auto is = [prev_action, curr_action](const auto &a, const auto &b) {
                return prev_action == a && curr_action == b;
            };
            const bool is_call_check = is(A::CALL, A::CHECK);
            const bool is_check_check = is(A::CHECK, A::CHECK);
            const bool is_raise_call = is(A::RAISE, A::CALL);
            if ((is_call_check || is_check_check || is_raise_call) && state.position > 0) {
                // This is the last card that was dealt after the previous round that ended
                // Note that this is incorrect for the first round of betting, but is only relevant
                // after the fourth betting round (after the river)
                const int last_card_idx = state.round + 1;
                const auto &last_card = history.common_cards.at(last_card_idx);
                const bool last_card_is_black =
                    last_card.value().suit == StandardDeck::Suits::SPADES ||
                    last_card.value().suit == StandardDeck::Suits::CLUBS;

                if (state.round >= 3 && last_card_is_black) {
                    state.is_game_over = true;
                    state.showdown_required = true;
                    return state;
                }

                state.position = 0;
                state.round++;
                did_advance_to_new_betting_round = true;
            }
        }
        if (!did_advance_to_new_betting_round) {
            state.position++;
        }
    }
    return state;
}

}  // namespace

std::ostream &operator<<(std::ostream &out, const RobPokerPlayer player) {
    out << wise_enum::to_string(player);
    return out;
}

std::ostream &operator<<(std::ostream &out, const RobPokerAction action) {
    out << wise_enum::to_string(action);
    return out;
}

std::optional<RobPokerPlayer> up_next(const RobPokerHistory &history) {
    using P = RobPokerPlayer;
    const bool are_common_cards_dealt = history.common_cards.front().has_value();
    const bool are_player_cards_dealt = history.hole_cards[P::PLAYER1].front().has_value() &&
                                        history.hole_cards[P::PLAYER2].front().has_value();

    if (!(are_common_cards_dealt && are_player_cards_dealt)) {
        return P::CHANCE;
    }

    const BettingState betting_state = compute_betting_state(history);
    if (betting_state.is_game_over) {
        return std::nullopt;
    }

    // Player 1 starts the first betting round, but player 2 starts each following round.
    const bool is_first_round = betting_state.round == 0;
    const bool is_even_position = betting_state.position % 2 == 0;
    return is_first_round ? (is_even_position ? P::PLAYER1 : P::PLAYER2)
                          : (is_even_position ? P::PLAYER2 : P::PLAYER1);
}

ChanceResult play(const RobPokerHistory &history, InOut<std::mt19937> gen) {
    static_assert(StandardDeck::NUM_CARDS >= std::tuple_size_v<decltype(history.common_cards)>);
    auto out = history;
    double probability;
    bool is_done_dealing = false;
    while (!is_done_dealing) {
        probability = 1.0;
        StandardDeck deck;
        deck.shuffle(gen);
        // Deal the player cards
        for (const auto player : {RobPokerPlayer::PLAYER1, RobPokerPlayer::PLAYER2}) {
            for (auto &card : out.hole_cards[player]) {
                probability *= 1.0 / deck.size();
                card =
                    RobPokerHistory::FogCard(deck.deal_card().value(), make_private_info(player));
            }
        }

        // Deal the common cards
        for (int i = 0; i < static_cast<int>(history.common_cards.size()); i++) {
            if (!is_done_dealing) {
                probability *= 1.0 / deck.size();
                out.common_cards[i] = RobPokerHistory::FogCard(
                    deck.deal_card().value(), make_private_info(RobPokerPlayer::CHANCE));

                // A deal is complete if at least 5 cards have been dealt and the last card isn't
                // red
                const bool is_last_card_black =
                    out.common_cards[i].value().suit == StandardDeck::Suits::CLUBS ||
                    out.common_cards[i].value().suit == StandardDeck::Suits::SPADES;
                const bool at_least_five_cards = i >= 4;
                is_done_dealing = is_last_card_black && at_least_five_cards;
            } else {
                out.common_cards[i] = {};
            }
        }
    }

    return {.history = out, .probability = probability};
}

RobPokerHistory play(const RobPokerHistory &history, const RobPokerAction &action) {
    auto out = history;
    out.actions.push_back(action);

    return out;
}

std::vector<RobPokerAction> possible_actions(const RobPokerHistory &history) {
    const BettingState betting_state = compute_betting_state(history);
    std::vector<RobPokerAction> out;
    if (betting_state.is_game_over) {
        // The game is over, nothing to do
        return out;
    }
    // Folding is always an option while the game is underway
    out.push_back(RobPokerAction::FOLD);

    // Raising is always an option
    // TODO take chip stack size into account
    out.push_back(RobPokerAction::RAISE);

    // Calling is only possible if the previous action in the current betting round was a raise or
    // it's the first action of the first round
    const bool is_first_action = history.actions.empty();
    const bool was_previous_raise =
        !history.actions.empty() && history.actions.back() == RobPokerAction::RAISE;
    const bool is_after_first_action_in_round = betting_state.position > 0;
    if (is_first_action || (is_after_first_action_in_round && was_previous_raise)) {
        out.push_back(RobPokerAction::CALL);
    }

    // You can't check after a raise
    if (!(was_previous_raise || is_first_action)) {
        out.push_back(RobPokerAction::CHECK);
    }
    return out;
}

std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player) {
    const auto betting_state = compute_betting_state(history);
    (void)player;
    if (!betting_state.is_game_over) {
        return std::nullopt;
    }
    const RobPokerPlayer opponent =
        player == RobPokerPlayer::PLAYER1 ? RobPokerPlayer::PLAYER2 : RobPokerPlayer::PLAYER1;

    const int pot_value = 1;

    if (betting_state.showdown_required) {
        const int player_hand_rank = evaluate_hand(history, player);
        const int opponent_hand_rank = evaluate_hand(history, opponent);
        if (player_hand_rank == opponent_hand_rank) {
            return 0;
        }
        const int sign = player_hand_rank > opponent_hand_rank ? 1 : -1;
        return sign * pot_value;
    }
    // determine which player folded and get sign
    // Player 1 starts the first round of betting, so if the betting position is even, player 1
    // folded
    // On later rounds, player 2 starts the betting, so if the betting position is even, player
    // 2 folded
    const auto folding_player =
        betting_state.round == 0
            ? (betting_state.position % 2 == 0 ? RobPokerPlayer::PLAYER1 : RobPokerPlayer::PLAYER2)
            : (betting_state.position % 2 == 0 ? RobPokerPlayer::PLAYER2 : RobPokerPlayer::PLAYER1);
    const int sign = player == folding_player ? -1 : 1;
    return sign * pot_value;
}
//
// RobPokerPoker::InfoSetId infoset_id_from_history(const RobPokerHistory &hist);
// RobPokerPoker::InfoSetId infoset_id_from_information(const RobPokerHistory::Card private_card,
//                                                      const std::vector<RobPokerAction> &actions);
std::string to_string(const RobPokerHistory &hist) {
    std::stringstream out;
    for (const auto player : {RobPokerPlayer::PLAYER1, RobPokerPlayer::PLAYER2}) {
        out << wise_enum::to_string(player) << ": ["
            << to_string(hist.hole_cards[player].at(0).value()) << ", "
            << to_string(hist.hole_cards[player].at(1).value()) << "] ";
    }

    out << "Common Cards: [";
    for (const auto &card : hist.common_cards) {
        if (card.has_value()) {
            out << to_string(card.value()) << ",";
        }
    }
    out.seekp(-1, std::ios_base::end);
    out << "]";

    return out.str();
}

int evaluate_hand(const RobPokerHistory &history, const RobPokerPlayer player) {
    const auto common_card_end_iter =
        std::find_if_not(history.common_cards.begin(), history.common_cards.end(),
                         [](const auto &card) { return card.has_value(); });
    const int common_card_end_idx =
        std::distance(history.common_cards.begin(), common_card_end_iter);
    constexpr int MAX_HAND_SIZE = 5;
    // Enumerate all possible 5 card hands for the current player and keep the max;
    int max_hand_value = std::numeric_limits<int>::lowest();
    std::array<int, MAX_HAND_SIZE> idxs = {-2, -1, 0, 1, 2};
    bool done = false;
    auto card_idx_from_card = [](const auto card) {
        const int card_idx = static_cast<int>(card.rank) * 4 + static_cast<int>(card.suit);
        return card_idx;
    };
    const omp::HandEvaluator evaluator;
    while (!done) {
        // Create the hand
        omp::Hand hand = omp::Hand::empty();
        for (const auto idx : idxs) {
            if (idx < 0) {
                hand += omp::Hand(
                    card_idx_from_card(history.hole_cards[player].at(std::abs(idx) - 1).value()));
            } else {
                hand += omp::Hand(card_idx_from_card(history.common_cards.at(idx).value()));
            }
        }

        // Evaluate the hand
        max_hand_value = std::max(max_hand_value, static_cast<int>(evaluator.evaluate(hand)));

        // Try to get the next indices
        bool did_update = false;
        for (int idx = 4; idx >= 0; idx--) {
            const int idx_from_end = idxs.size() - idx;
            const int max_for_curr_idx = common_card_end_idx - idx_from_end;
            if (idxs[idx] < max_for_curr_idx) {
                idxs[idx]++;
                for (int future_idx = idx + 1; future_idx < static_cast<int>(idxs.size());
                     future_idx++) {
                    idxs[future_idx] = idxs[future_idx - 1] + 1;
                }
                did_update = true;
                break;
            }
        }
        // If we can't, we're done
        done = !did_update;
    }
    return max_hand_value;
}

}  // namespace robot::domain
