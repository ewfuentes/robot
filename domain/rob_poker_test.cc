
#include "domain/rob_poker.hh"

#include <random>

#include "domain/deck.hh"
#include "gtest/gtest.h"

namespace robot::domain {
namespace {
using Action = RobPokerAction;
using Suits = StandardDeck::Suits;
using Ranks = StandardDeck::Ranks;
using SCard = StandardDeck::Card;

auto make_fog_card(const StandardDeck::Card card,
                   const RobPokerPlayer owner = RobPokerPlayer::CHANCE) {
    return RobPokerHistory::FogCard(card, make_private_info(owner));
}

std::string make_test_name(const auto &test_param_info) {
    std::stringstream out;
    const auto &test_input = test_param_info.param;
    out << (test_input.first.empty() ? "none_" : "");
    for (const auto action : test_input.first) {
        out << wise_enum::to_string(action) << "_";
    }
    out << "yields_"
        << (test_input.second.has_value() ? wise_enum::to_string(test_input.second.value())
                                          : "none");
    return out.str();
}
}  // namespace

class RobPokerUpNextNoRunTest : public testing::Test,
                                public testing::WithParamInterface<
                                    std::pair<std::vector<Action>, std::optional<RobPokerPlayer>>> {
};

TEST_P(RobPokerUpNextNoRunTest, test_up_next) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{.rank = Ranks::_2, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_3, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_4, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_5, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_6, .suit = Suits::SPADES}),
            },
        .actions = actions,
    };

    // Action
    const auto maybe_next_player = up_next(history);

    // Verification
    ASSERT_EQ(maybe_next_player.has_value(), expected_value.has_value());
    if (expected_value.has_value()) {
        EXPECT_EQ(maybe_next_player.value(), expected_value.value());
    }
}

std::pair<std::vector<Action>, std::optional<RobPokerPlayer>> up_next_no_run_test_cases[] = {
    {{}, RobPokerPlayer::PLAYER1},
    {{Action::CALL}, RobPokerPlayer::PLAYER2},
    {{Action::RAISE}, RobPokerPlayer::PLAYER2},
    {{Action::FOLD}, {}},
    {{Action::CALL, Action::CHECK}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::RAISE}, RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::RAISE, Action::RAISE}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK}, RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK},
     RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK},
     {}},
};
INSTANTIATE_TEST_SUITE_P(UpNext, RobPokerUpNextNoRunTest,
                         testing::ValuesIn(up_next_no_run_test_cases),
                         [](const auto &info) { return make_test_name(info); });

class RobPokerUpNextWithRunTest : public RobPokerUpNextNoRunTest {};

TEST_P(RobPokerUpNextWithRunTest, test_up_next) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{.rank = Ranks::_2, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_3, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_4, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_5, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_6, .suit = Suits::HEARTS}),
                make_fog_card(SCard{.rank = Ranks::_7, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_8, .suit = Suits::CLUBS}),
            },
        .actions = actions,
    };

    // Action
    const auto maybe_next_player = up_next(history);

    // Verification
    ASSERT_EQ(maybe_next_player.has_value(), expected_value.has_value());
    if (expected_value.has_value()) {
        EXPECT_EQ(maybe_next_player.value(), expected_value.value());
    }
}

std::pair<std::vector<Action>, std::optional<RobPokerPlayer>> up_next_with_run_test_cases[] = {
    {{}, RobPokerPlayer::PLAYER1},
    {{Action::CALL}, RobPokerPlayer::PLAYER2},
    {{Action::RAISE}, RobPokerPlayer::PLAYER2},
    {{Action::FOLD}, {}},
    {{Action::CALL, Action::CHECK}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::RAISE}, RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::RAISE, Action::RAISE}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK}, RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK}, RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK},
     RobPokerPlayer::PLAYER1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK},
     RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
     RobPokerPlayer::PLAYER2},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
       Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
     {}},
};
INSTANTIATE_TEST_SUITE_P(UpNext, RobPokerUpNextWithRunTest,
                         testing::ValuesIn(up_next_with_run_test_cases),
                         [](const auto &info) { return make_test_name(info); });

// TEST(RobPokerTest, test_game) {
//     // Setup
//     std::mt19937 gen(0);
//     RobPokerHistory history;
//
//     // Action
//     auto maybe_current_player = up_next(history);
//     while (maybe_current_player.has_value()) {
//         const auto current_player = maybe_current_player.value();
//         if (current_player == RobPokerPlayer::CHANCE) {
//             const auto chance_result = play(history, make_in_out(gen));
//             history = chance_result.history;
//         } else {
//             const auto actions = possible_actions(history);
//             for (const auto desired_action : {RobPokerAction::CHECK, RobPokerAction::CALL}) {
//                 const auto iter = std::find(actions.begin(), actions.end(), desired_action);
//                 if (iter != actions.end()) {
//                     history = play(history, desired_action);
//                     break;
//                 }
//             }
//         }
//         maybe_current_player = up_next(history);
//     }
//
//     // Verification
//     const auto maybe_terminal_value = terminal_value(history, RobPokerPlayer::PLAYER1);
//     EXPECT_TRUE(maybe_terminal_value.has_value());
// }

}  // namespace robot::domain
