
#include "domain/rob_poker.hh"

#include <ios>
#include <iostream>
#include <sstream>

#include "domain/deck.hh"
#include "domain/fog.hh"

namespace robot::domain {

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
    using A = RobPokerAction;
    const bool are_common_cards_dealt = history.common_cards.front().has_value();
    const bool are_player_cards_dealt = history.hole_cards[P::PLAYER1].front().has_value() &&
                                        history.hole_cards[P::PLAYER2].front().has_value();

    if (!(are_common_cards_dealt && are_player_cards_dealt)) {
        return P::CHANCE;
    } else if (history.actions.empty()) {
        return P::PLAYER1;
    } else if (history.actions.back() == A::FOLD) {
        return std::nullopt;
    }

    P next_player = P::PLAYER2;
    int curr_betting_round = 0;
    bool is_first_betting_action = false;
    for (int i = 1; i < static_cast<int>(history.actions.size()); i++) {
        const auto prev_action = history.actions.at(i - 1);
        const auto curr_action = history.actions.at(i);
        const auto is = [prev_action, curr_action](const auto &a, const auto &b) {
            return prev_action == a && curr_action == b;
        };
        const bool is_call_check = is(A::CALL, A::CHECK);
        const bool is_check_check = is(A::CHECK, A::CHECK);
        const bool is_raise_call = is(A::RAISE, A::CALL);
        if ((is_call_check || is_check_check || is_raise_call) && !is_first_betting_action) {
            curr_betting_round++;
            is_first_betting_action = true;

            // Player 2 always starts every betting round after the first
            next_player = P::PLAYER2;

            // unless it's the terminal betting round
            const int last_card_idx = curr_betting_round;
            const auto &last_card = history.common_cards.at(last_card_idx);
            const bool last_card_is_black = last_card.value().suit == StandardDeck::Suits::SPADES ||
                                            last_card.value().suit == StandardDeck::Suits::CLUBS;

            if (curr_betting_round > 3 && last_card_is_black) {
                return std::nullopt;
            }
        } else {
            is_first_betting_action = false;
            next_player = next_player == P::PLAYER1 ? P::PLAYER2 : P::PLAYER1;
        }
    }
    return next_player;
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
    (void)history;
    return {RobPokerAction::CHECK, RobPokerAction::CALL, RobPokerAction::RAISE};
}

std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player) {
    (void)history;
    (void)player;
    return 0;
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

}  // namespace robot::domain
