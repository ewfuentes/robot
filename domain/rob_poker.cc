
#include "domain/rob_poker.hh"

#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <variant>

#include "common/indexed_array.hh"
#include "domain/deck.hh"
#include "domain/fog.hh"
#include "omp/Hand.h"
#include "omp/HandEvaluator.h"

namespace robot::domain {
BettingState compute_betting_state(const RobPokerHistory &history) {
    BettingState state = {.is_game_over = false,
                          .showdown_required = false,
                          .round = 0,
                          .position = 0,
                          .put_in_pot = {0}};
    if (history.actions.empty()) {
        return state;
    }

    RobPokerPlayer current_player = RobPokerPlayer::PLAYER1;
    for (int i = 0; i < static_cast<int>(history.actions.size()); i++) {
        RobPokerPlayer other_player = current_player == RobPokerPlayer::PLAYER1
                                          ? RobPokerPlayer::PLAYER2
                                          : RobPokerPlayer::PLAYER1;
        const auto curr_action = history.actions.at(i);

        // Update the bet amounts
        std::visit(
            [&state, &current_player, &other_player](const auto &action) mutable {
                using T = std::decay_t<decltype(action)>;
                const int continue_cost =
                    state.put_in_pot[other_player] - state.put_in_pot[current_player];
                if constexpr (std::is_same_v<T, RaiseAction>) {
                    state.put_in_pot[current_player] += continue_cost + action.amount;
                } else if constexpr (std::is_same_v<T, RaisePotAction>) {
                    const int current_pot =
                        state.put_in_pot[current_player] + state.put_in_pot[other_player];
                    state.put_in_pot[current_player] = continue_cost + current_pot;
                } else if constexpr (std::is_same_v<T, AllInAction>) {
                    state.put_in_pot[current_player] = RobPokerHistory::STARTING_STACK_SIZE;
                } else if constexpr (std::is_same_v<T, CallAction>) {
                    state.put_in_pot[current_player] += continue_cost;
                }
            },
            curr_action);

        bool did_advance_to_new_betting_round = false;
        if (std::holds_alternative<FoldAction>(curr_action)) {
            state.is_game_over = true;
            return state;
        } else if (i > 0) {
            const auto prev_action = history.actions.at(i - 1);
            const auto is = [prev_action, curr_action](auto a, auto b) {
                return std::holds_alternative<decltype(a)>(prev_action) &&
                       std::holds_alternative<decltype(b)>(curr_action);
            };
            const bool is_call_check = is(CallAction{}, CheckAction{});
            const bool is_check_check = is(CheckAction{}, CheckAction{});
            const bool is_raise_call =
                is(RaiseAction{}, CallAction{}) || is(RaisePotAction{}, CallAction{});
            const bool is_opening_call = is_raise_call && i == 2;
            const bool is_allin_call = is(AllInAction{}, CallAction{});
            if ((is_call_check || is_check_check || is_raise_call) && state.position > 0 &&
                !is_opening_call) {
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
                current_player = RobPokerPlayer::PLAYER2;
                state.round++;
                did_advance_to_new_betting_round = true;
            } else if (is_allin_call) {
                state.is_game_over = true;
                state.showdown_required = true;
                return state;
            }
        }
        if (!did_advance_to_new_betting_round) {
            state.position++;
            current_player = other_player;
        }
    }
    return state;
}

std::ostream &operator<<(std::ostream &out, const RobPokerPlayer player) {
    out << wise_enum::to_string(player);
    return out;
}

std::string to_string(const RobPokerAction &action) {
    return std::visit(
        [](auto &&action) -> std::string {
            using T = std::decay_t<decltype(action)>;
            if constexpr (std::is_same_v<T, RaiseAction>) {
                return "Raise" + std::to_string(action.amount);
            } else {
                return std::string(T::name);
            }
        },
        action);
}

std::ostream &operator<<(std::ostream &out, const RobPokerAction action) {
    out << to_string(action);
    return out;
}

std::ostream &operator<<(std::ostream &out, const RobPokerHistory &history) {
    out << to_string(history);
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

    // It feels icky to do this here, but post the blinds
    if (out.actions.empty()) {
        out.actions.push_back(RaiseAction{RobPokerHistory::SMALL_BLIND});
        out.actions.push_back(
            RaiseAction{RobPokerHistory::BIG_BLIND - RobPokerHistory::SMALL_BLIND});
    }

    return {.history = out, .probability = probability};
}

RobPokerHistory play(const RobPokerHistory &history, const RobPokerAction &action) {
    auto out = history;
    out.actions.push_back(action);
    const BettingState betting_state = compute_betting_state(out);

    // If this is the start of a new round, update the visibility of the cards
    if (betting_state.is_game_over && betting_state.showdown_required) {
        for (auto &card : out.common_cards) {
            if (card.has_value()) {
                card.update_visibility([](const RobPokerPlayer) { return true; });
            }
        }
    } else if (betting_state.position == 0) {
        if (betting_state.round == 1) {
            // Show the flop
            for (int i = 0; i < 3; i++) {
                out.common_cards[i].update_visibility([](const RobPokerPlayer) { return true; });
            }
        } else if (betting_state.round > 1) {
            // Show each subsequent card
            if (out.common_cards[betting_state.round + 1].has_value()) {
                out.common_cards[betting_state.round + 1].update_visibility(
                    [](const RobPokerPlayer) { return true; });
            }
        }
    }
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
    out.push_back(FoldAction{});

    const int most_contributed = std::max(betting_state.put_in_pot[RobPokerPlayer::PLAYER1],
                                          betting_state.put_in_pot[RobPokerPlayer::PLAYER2]);

    const int max_raise = RobPokerHistory::STARTING_STACK_SIZE - most_contributed;
    if (max_raise > 0) {
        out.push_back(RaiseAction{max_raise});
    }

    // Calling is only possible if the previous action in the current betting round was a raise
    const bool was_previous_raise =
        std::holds_alternative<RaiseAction>(history.actions.back()) ||
        std::holds_alternative<RaisePotAction>(history.actions.back()) ||
        std::holds_alternative<AllInAction>(history.actions.back());
    if (was_previous_raise) {
        out.push_back(CallAction{});
    }

    // You can't check after a raise
    if (!was_previous_raise) {
        out.push_back(CheckAction{});
    }
    return out;
}

std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player) {
    const auto betting_state = compute_betting_state(history);
    if (!betting_state.is_game_over) {
        return std::nullopt;
    }
    const RobPokerPlayer opponent =
        player == RobPokerPlayer::PLAYER1 ? RobPokerPlayer::PLAYER2 : RobPokerPlayer::PLAYER1;

    if (betting_state.showdown_required) {
        const int player_hand_rank = evaluate_hand(history, player);
        const int opponent_hand_rank = evaluate_hand(history, opponent);
        if (player_hand_rank == opponent_hand_rank) {
            return 0;
        }
        return player_hand_rank > opponent_hand_rank ? betting_state.put_in_pot[opponent]
                                                     : -betting_state.put_in_pot[player];
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
    return player == folding_player ? -betting_state.put_in_pot[player]
                                    : betting_state.put_in_pot[opponent];
}
//
// RobPokerPoker::InfoSetId infoset_id_from_history(const RobPokerHistory &hist);
// RobPokerPoker::InfoSetId infoset_id_from_information(const RobPokerHistory::Card private_card,
//                                                      const std::vector<RobPokerAction> &actions);
std::string to_string(const RobPokerHistory &hist) {
    std::stringstream out;
    for (const auto player : {RobPokerPlayer::PLAYER1, RobPokerPlayer::PLAYER2}) {
        out << wise_enum::to_string(player) << ": [";
        if (hist.hole_cards[player][0].has_value()) {
            out << to_string(hist.hole_cards[player].at(0).value()) << ", "
                << to_string(hist.hole_cards[player].at(1).value());
        }
        out << "] ";
    }

    out << "Common Cards: [";
    {
        bool did_print = false;
        for (const auto &card : hist.common_cards) {
            if (card.has_value() && card.is_visible_to(RobPokerPlayer::PLAYER1)) {
                out << to_string(card.value()) << ",";
                did_print = true;
            }
        }
        if (did_print) {
            out.seekp(-1, std::ios_base::end);
        }
    }
    out << "]";

    out << " Actions: [";
    {
        bool did_print = false;
        for (const auto &action : hist.actions) {
            out << action << ",";
            did_print = true;
        }
        if (did_print) {
            out.seekp(-1, std::ios_base::end);
        }
    }
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
