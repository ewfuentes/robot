
#include <cstdint>
#include <iostream>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/fog.hh"
#include "domain/rob_poker.hh"
#include "omp/CardRange.h"
#include "omp/EquityCalculator.h"
#include "pybind11/pybind11.h"
#include "vector"

namespace robot::experimental::pokerbots {
namespace {
using Card = domain::StandardDeck::Card;

struct EquityResult {
    double equity;
    uint64_t num_evaluations;
};

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

std::tuple<domain::RobPokerHistory, domain::StandardDeck> create_history_and_deck_from_known_cards(
    const std::vector<Card> &hole_cards, const std::vector<Card> &board_cards) {
    constexpr auto PLAYER = domain::RobPokerPlayer::PLAYER1;

    domain::RobPokerHistory current_state;
    domain::StandardDeck deck;
    auto end_iter = deck.end();
    for (int i = 0; i < static_cast<int>(hole_cards.size()); i++) {
        current_state.hole_cards[PLAYER][i] =
            domain::RobPokerHistory::FogCard(hole_cards[i], make_private_info(PLAYER));
        end_iter = std::remove_if(deck.begin(), end_iter, [&](const auto &deck_card) {
            return deck_card == hole_cards[i];
        });
    }
    for (int i = 0; i < static_cast<int>(board_cards.size()); i++) {
        current_state.common_cards[i] =
            domain::RobPokerHistory::FogCard(board_cards[i], make_private_info(PLAYER));
        end_iter = std::remove_if(deck.begin(), end_iter, [&](const auto &deck_card) {
            return deck_card == board_cards[i];
        });
    }

    deck.erase(end_iter);

    return std::make_tuple(std::move(current_state), std::move(deck));
}

EquityResult evaluate_hand(const std::string &hand, const std::string &opponent_hand_str,
                           const std::string &board_str, const double timeout_s) {
    const std::vector<omp::CardRange> ranges = {{hand}, {opponent_hand_str}};
    const uint64_t board = omp::CardRange::getCardMask(board_str);
    omp::EquityCalculator calculator;
    calculator.setTimeLimit(timeout_s);
    calculator.start(ranges, board);
    calculator.wait();
    const auto results = calculator.getResults();
    return {
        .equity = results.equity[0],
        .num_evaluations = results.hands,
    };
}

EquityResult evaluate_hand_potential(const std::string &hand_str, const std::string &board_str,
                                     const double timeout_s) {
    const auto hole_cards = cards_from_string(hand_str);
    const auto board_cards = cards_from_string(board_str);
    const auto [initial_state, remaining_deck] =
        create_history_and_deck_from_known_cards(hole_cards, board_cards);

    const time::RobotTimestamp start = time::current_robot_time();
    const time::RobotTimestamp end_time = start + time::as_duration(timeout_s);
    int num_evals = 0;
    std::mt19937 gen(start.time_since_epoch().count());
    // counts[b][a] keeps track of results before and after showing
    // b = before, a = after
    // 0 = behind, 1 = tied, 2 = ahead
    std::array<std::array<int, 3>, 3> counts{0};
    const auto get_rank_idx = [](const int a, const int b) { return a == b ? 0 : (a < b ? 0 : 2); };

    const int before_player_rank =
        domain::evaluate_hand(initial_state, domain::RobPokerPlayer::PLAYER1);
    while (time::current_robot_time() < end_time) {
        num_evals++;
        auto sample_history = initial_state;
        auto deck = remaining_deck;
        deck.shuffle(make_in_out(gen));

        // Deal two cards to the second player
        for (auto &hole_card : sample_history.hole_cards[domain::RobPokerPlayer::PLAYER2]) {
            hole_card = domain::RobPokerHistory::FogCard(
                deck.deal_card().value(), make_private_info(domain::RobPokerPlayer::PLAYER2));
        }

        // Evaluate hands
        const int before_opponent_rank =
            domain::evaluate_hand(sample_history, domain::RobPokerPlayer::PLAYER2);
        const int before_idx = get_rank_idx(before_player_rank, before_opponent_rank);

        // Deal remaining board
        bool is_done_dealing = false;
        for (int i = 0; i < static_cast<int>(sample_history.common_cards.size()); i++) {
            if (!is_done_dealing) {
                if (!sample_history.common_cards[i].has_value()) {
                    sample_history.common_cards[i] = domain::RobPokerHistory::FogCard(
                        deck.deal_card().value(),
                        make_private_info(domain::RobPokerPlayer::CHANCE));
                }

                // A deal is complete if at least 5 cards have been dealt and the last card isn't
                // red
                const bool is_last_card_black = sample_history.common_cards[i].value().suit ==
                                                    domain::StandardDeck::Suits::CLUBS ||
                                                sample_history.common_cards[i].value().suit ==
                                                    domain::StandardDeck::Suits::SPADES;
                const bool at_least_five_cards = i >= 4;
                is_done_dealing = is_last_card_black && at_least_five_cards;
            } else {
                sample_history.common_cards[i] = {};
            }
        }

        // Evaluate hands
        const int after_player_rank =
            domain::evaluate_hand(sample_history, domain::RobPokerPlayer::PLAYER1);
        const int after_opponent_rank =
            domain::evaluate_hand(sample_history, domain::RobPokerPlayer::PLAYER2);
        const int after_idx = get_rank_idx(after_player_rank, after_opponent_rank);
        counts[before_idx][after_idx]++;
    }

    const auto sum_row = [&](const int row) {
        return counts[row][0] + counts[row][1] + counts[row][2];
    };

    const int before_behind_counts = sum_row(0);
    const int before_tied_counts = sum_row(1);
    const int before_ahead_counts = sum_row(2);
    (void)before_behind_counts;
    const double hand_strength =
        static_cast<double>(before_ahead_counts + before_tied_counts / 2.0) / num_evals;

    const double positive_potential =
        static_cast<double>(counts[0][2] + (counts[0][1] + counts[1][2]) / 2.0) / num_evals;

    const double negative_potential =
        static_cast<double>(counts[2][0] + (counts[2][1] + counts[1][0]) / 2.0) / num_evals;

    const double hand_potential =
        (1.0 - hand_strength) * positive_potential + hand_strength * (1.0 - negative_potential);
    std::cout << "Counts for hole cards: " << hand_str << " board: " << board_str << std::endl;
    for (const auto &row : counts) {
        for (const auto &item : row) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "HS: " << hand_strength << " p pot: " << positive_potential
              << " n pot: " << negative_potential << " hp: " << hand_potential << std::endl;

    return {
        .equity = hand_potential,
        .num_evaluations = static_cast<uint64_t>(num_evals),
    };
}
}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(hand_evaluator_python, m) {
    m.doc() = "Hand Evaluator";

    py::class_<EquityResult>(m, "EquityResult")
        .def_readonly("equity", &EquityResult::equity)
        .def_readonly("num_evaluations", &EquityResult::num_evaluations);

    m.def("evaluate_hand", &evaluate_hand);
    m.def("evaluate_hand_potential", &evaluate_hand_potential);
}
}  // namespace robot::experimental::pokerbots
