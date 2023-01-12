
#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "omp/CardRange.h"
#include "omp/EquityCalculator.h"
#include "pybind11/pybind11.h"
#include "vector"

namespace robot::experimental::pokerbots {
namespace {
struct EquityResult {
    double equity;
    int num_evaluations;
};

EquityResult evaluate_hand(const std::string &hand, const std::string &opponent_hand_str,
                           const std::string &board_str, const double timeout_s) {
    (void)opponent_hand_str;
    constexpr auto PLAYER = domain::RobPokerPlayer::PLAYER1;
    const std::string ranks = "23456789TJQKA";
    const std::string suits = "shcd";
    const auto parse_str = [&](const std::string &cards) {
        std::vector<domain::StandardDeck::Card> out;
        for (int i = 0; i < static_cast<int>(cards.size()); i += 2) {
            out.push_back({
                .rank = static_cast<domain::StandardRanks>(ranks.find(cards[i])),
                .suit = static_cast<domain::StandardSuits>(suits.find(cards[i + 1])),
            });
        }
        return out;
    };
    const auto hole_cards = parse_str(hand);
    const auto board_cards = parse_str(board_str);
    domain::RobPokerHistory current_state;
    for (int i = 0; i < static_cast<int>(hole_cards.size()); i++) {
        current_state.hole_cards[PLAYER][i] =
            domain::RobPokerHistory::FogCard(hole_cards[i], make_private_info(PLAYER));
    }
    for (int i = 0; i < static_cast<int>(board_cards.size()); i++) {
        current_state.common_cards[i] =
            domain::RobPokerHistory::FogCard(board_cards[i], make_private_info(PLAYER));
    }

    const time::RobotTimestamp start = time::current_robot_time();
    const time::RobotTimestamp end_time = start + time::as_duration(timeout_s);
    int num_evals = 0;
    std::mt19937 gen(start.time_since_epoch().count());
    // counts[b][a] keeps track of results before and after showing
    // b = before, a = after
    // 0 = behind, 1 = tied, 2 = ahead
    std::array<std::array<int, 3>, 3> counts;

    while (time::current_robot_time() < end_time) {
      num_evals++;
      const auto new_hist = play(current_state, make_in_out(gen)).history;
      // Evaluate hands before and after

    }

    (void) counts;
    return {
        .equity = 0.0,
        .num_evaluations = num_evals,
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
}
}  // namespace robot::experimental::pokerbots
