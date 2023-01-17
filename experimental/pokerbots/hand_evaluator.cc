
#include "experimental/pokerbots/hand_evaluator.hh"

#include <iostream>
#include <limits>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "omp/CardRange.h"
#include "omp/EquityCalculator.h"

namespace robot::experimental::pokerbots {
using Card = domain::StandardDeck::Card;

std::vector<Card> cards_from_string(const std::string &cards_str) {
    const std::string ranks = "23456789TJQKA";
    const std::string suits = "shcd";
    std::vector<Card> out;
    for (int i = 0; i < static_cast<int>(cards_str.size()); i += 2) {
        out.push_back({
            .rank = static_cast<domain::StandardRanks>(ranks.find(cards_str[i])),
            .suit = static_cast<domain::StandardSuits>(suits.find(cards_str[i + 1])),
        });
    }
    return out;
}

template <typename T1, typename T2>
domain::RobPokerHistory create_history_from_known_cards(const T1 &hole_cards,
                                                        const T2 &board_cards) {
    constexpr auto PLAYER = domain::RobPokerPlayer::PLAYER1;
    domain::RobPokerHistory current_state;
    for (int i = 0; i < static_cast<int>(hole_cards.size()); i++) {
        current_state.hole_cards[PLAYER][i] =
            domain::RobPokerHistory::FogCard(hole_cards[i], make_private_info(PLAYER));
    }
    for (int i = 0; i < static_cast<int>(board_cards.size()); i++) {
        current_state.common_cards[i] =
            domain::RobPokerHistory::FogCard(board_cards[i], [](const auto) { return true; });
    }

    return current_state;
}

domain::StandardDeck deck_from_history(const domain::RobPokerHistory &history,
                                       const domain::RobPokerPlayer player) {
    domain::StandardDeck deck;
    auto end_iter = deck.end();
    for (const auto &card : history.hole_cards[player]) {
        end_iter = std::remove_if(deck.begin(), end_iter,
                                  [&](const auto &deck_card) { return deck_card == card.value(); });
    }
    for (const auto &card : history.common_cards) {
        end_iter = std::remove_if(deck.begin(), end_iter, [&](const auto &deck_card) {
            return card.has_value() && deck_card == card.value();
        });
    }

    deck.erase(end_iter);
    return deck;
}

ExpectedStrengthResult evaluate_expected_strength(const std::string &hand,
                                                  const std::string &opponent_hand_str,
                                                  const std::string &board_str,
                                                  const std::optional<double> timeout_s,
                                                  const std::optional<int> num_hands) {
    const std::vector<omp::CardRange> ranges = {{hand}, {opponent_hand_str}};
    const uint64_t board = omp::CardRange::getCardMask(board_str);
    omp::EquityCalculator calculator;
    if (timeout_s.has_value()) {
        calculator.setTimeLimit(timeout_s.value());
    }
    if (num_hands.has_value()) {
        calculator.setHandLimit(num_hands.value());
    }
    calculator.start(ranges, board);
    calculator.wait();
    const auto results = calculator.getResults();
    return {
        .strength = results.equity[0],
        .num_evaluations = results.hands,
    };
}

std::array<uint64_t, 33> eval_strength_counts = {0};
StrengthPotentialResult evaluate_strength_potential(
    const domain::RobPokerHistory &history, const domain::RobPokerPlayer player,
    const std::optional<time::RobotTimestamp::duration> timeout, const std::optional<int> num_hands,
    InOut<std::mt19937> gen) {
    const auto remaining_deck = deck_from_history(history, player);
    const auto eval_time = timeout.value_or(time::RobotTimestamp::duration::max());
    const time::RobotTimestamp start = time::current_robot_time();
    const int hand_limit = num_hands.value_or(std::numeric_limits<int>::max());
    int num_evals = 0;

    // Create a container for all of the cards
    // The first two cards are the hole cards for the player of interest
    // The remaining cards are the board cards
    int num_cards = 0;
    std::array<domain::StandardDeck::Card, 33> workspace;
    const std::array<domain::StandardDeck::Card, 2> player_cards = {
        history.hole_cards[player][0].value(),
        history.hole_cards[player][1].value(),
    };

    auto populate_hole_cards = [&num_cards, &workspace](const auto &hole_cards) mutable {
        workspace[0] = hole_cards[0];
        workspace[1] = hole_cards[1];
        if (num_cards < 2) {
            num_cards = 2;
        }
    };

    auto push_card = [&num_cards, &workspace](const auto &card) mutable {
        workspace[num_cards++] = card;
    };

    auto pop_back = [&num_cards](const int num_items = 1) mutable { num_cards -= num_items; };

    populate_hole_cards(player_cards);
    for (const auto &fog_card : history.common_cards) {
        if (fog_card.has_value() && fog_card.is_visible_to(player)) {
            push_card(fog_card.value());
        }
    }

    // counts[b][a] keeps track of results before and after showing
    // b = before, a = after
    // 0 = behind, 1 = tied, 2 = ahead
    std::array<int, 9> counts{0};
    const auto get_rank_idx = [](const int a, const int b) { return a == b ? 0 : (a < b ? 0 : 2); };
    const auto flat_idx = [](const int a, const int b) { return a * 3 + b; };

    eval_strength_counts[num_cards]++;

    const int before_player_rank = domain::evaluate_hand(workspace, num_cards);
    while (time::current_robot_time() - start < eval_time && num_evals < hand_limit) {
        num_evals++;
        auto deck = remaining_deck;
        deck.shuffle(gen);

        // Deal two cards to the second player
        const std::array<domain::StandardDeck::Card, 2> opponent_cards = {deck.deal_card().value(),
                                                                          deck.deal_card().value()};

        // Evaluate hands
        populate_hole_cards(opponent_cards);
        const int before_opponent_rank = domain::evaluate_hand(workspace, num_cards);
        const int before_idx = get_rank_idx(before_player_rank, before_opponent_rank);

        // Deal remaining board
        bool is_done_dealing = false;
        int cards_dealt = 0;
        for (int i = num_cards; i < static_cast<int>(workspace.size()); i++) {
            if (!is_done_dealing) {
                cards_dealt++;
                push_card(deck.deal_card().value());
                // A deal is complete if at least 5 cards have been dealt and the last card isn't
                // red
                const bool is_last_card_black =
                    workspace[num_cards - 1].suit == domain::StandardDeck::Suits::CLUBS ||
                    workspace[num_cards - 1].suit == domain::StandardDeck::Suits::SPADES;
                const bool at_least_board_five_cards = num_cards >= 7;
                is_done_dealing = is_last_card_black && at_least_board_five_cards;
            }
        }

        if (cards_dealt > 4) {
          // Too many cards dealt, not worth evaluating
          num_evals--;
          pop_back(cards_dealt);
          continue;
        }

        // Evaluate hands
        const int after_opponent_rank = domain::evaluate_hand(workspace, num_cards);
        populate_hole_cards(player_cards);
        const int after_player_rank = domain::evaluate_hand(workspace, num_cards);
        const int after_idx = get_rank_idx(after_player_rank, after_opponent_rank);
        counts[flat_idx(before_idx, after_idx)]++;
        pop_back(cards_dealt);
    }

    const auto sum_row = [&](const int row) {
        return counts[flat_idx(row, 0)] + counts[flat_idx(row, 1)] + counts[flat_idx(row, 2)];
    };

    const int before_behind_counts = sum_row(0);
    const int before_tied_counts = sum_row(1);
    const int before_ahead_counts = sum_row(2);
    (void)before_behind_counts;
    const double hand_strength =
        static_cast<double>(before_ahead_counts + before_tied_counts / 2.0) / num_evals;

    const double positive_potential =
        static_cast<double>(counts[flat_idx(0, 2)] +
                            (counts[flat_idx(0, 1)] + counts[flat_idx(1, 2)]) / 2.0) /
        num_evals;

    const double negative_potential =
        static_cast<double>(counts[flat_idx(2, 0)] +
                            (counts[flat_idx(2, 1)] + counts[flat_idx(1, 0)]) / 2.0) /
        num_evals;

    const double hand_potential =
        (1.0 - hand_strength) * positive_potential + hand_strength * (1.0 - negative_potential);

    return {
        .strength = hand_strength,
        .strength_potential = hand_potential,
        .positive_potential = positive_potential,
        .negative_potential = negative_potential,
        .num_evaluations = static_cast<uint64_t>(num_evals),
    };
}

StrengthPotentialResult evaluate_strength_potential(
    const std::array<domain::StandardDeck::Card, 2> &hand,
    const std::vector<domain::StandardDeck::Card> &board,
    const std::optional<time::RobotTimestamp::duration> timeout,
    const std::optional<int> num_hands) {
    const auto history = create_history_from_known_cards(hand, board);
    std::mt19937 gen(0);
    return evaluate_strength_potential(history, domain::RobPokerPlayer::PLAYER1, timeout, num_hands,
                                       make_in_out(gen));
}

StrengthPotentialResult evaluate_strength_potential(const std::string &hand_str,
                                                    const std::string &board_str,
                                                    const std::optional<double> timeout_s,
                                                    const std::optional<int> num_hands) {
    const auto hole_cards = cards_from_string(hand_str);
    const auto board_cards = cards_from_string(board_str);
    const auto history = create_history_from_known_cards(hole_cards, board_cards);
    std::mt19937 gen(0);
    const auto timeout = timeout_s.has_value()
                             ? std::make_optional(time::as_duration(timeout_s.value()))
                             : std::nullopt;
    return evaluate_strength_potential(history, domain::RobPokerPlayer::PLAYER1, timeout, num_hands,
                                       make_in_out(gen));
}
}  // namespace robot::experimental::pokerbots
