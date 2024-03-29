
#include "domain/rob_poker.hh"

#include <random>
#include <variant>

#include "domain/deck.hh"
#include "gtest/gtest.h"

namespace robot::domain {
namespace {
using Action = RobPokerAction;
using Suits = StandardDeck::Suits;
using Ranks = StandardDeck::Ranks;
using SCard = StandardDeck::Card;

constexpr RaiseAction SMALL_BLIND = RaiseAction{RobPokerHistory::SMALL_BLIND};
constexpr RaiseAction BIG_BLIND =
    RaiseAction{RobPokerHistory::BIG_BLIND - RobPokerHistory::SMALL_BLIND};
constexpr RaiseAction RAISE = RaiseAction{20};

auto make_fog_card(const StandardDeck::Card card,
                   const RobPokerPlayer owner = RobPokerPlayer::CHANCE) {
    return RobPokerHistory::FogCard(card, make_private_info(owner));
}

auto make_public_card(const StandardDeck::Card card) {
    return RobPokerHistory::FogCard(card, [](const auto &) { return true; });
}

std::string range_to_string(const auto &range) {
    std::stringstream out;
    out << (range.empty() ? "none" : "");
    for (const auto &item : range) {
        out << item;
        if (&item != &range.back()) {
            out << "_";
        }
    }
    return out.str();
}

std::string optional_to_string(const auto &maybe_item) {
    return std::string(maybe_item.has_value() ? wise_enum::to_string(maybe_item.value()) : "none");
}

std::string optional_int_to_string(const auto &maybe_item) {
    if (!maybe_item.has_value()) {
        return "none";
    }
    const int item = maybe_item.value();
    return (item < 0 ? "neg" : "") + std::to_string(std::abs(item));
}

std::string make_test_name_range_to_optional(const auto &test_param_info) {
    const auto &test_input = test_param_info.param;
    return range_to_string(test_input.first) + "_yields_" + optional_to_string(test_input.second);
}

std::string make_test_name_range_to_optional_int(const auto &test_param_info) {
    const auto &test_input = test_param_info.param;
    return range_to_string(test_input.first) + "_yields_" +
           optional_int_to_string(test_input.second);
}

std::string make_test_name_range_to_range(const auto &test_param_info) {
    const auto &test_input = test_param_info.param;
    return range_to_string(test_input.first) + "_yields_" + range_to_string(test_input.second);
}
}  // namespace

class RobPokerUpNextNoRunTest : public testing::Test,
                                public testing::WithParamInterface<
                                    std::pair<std::vector<Action>, std::optional<RobPokerPlayer>>> {
};

TEST_P(RobPokerUpNextNoRunTest, test_up_next) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}, RobPokerPlayer::PLAYER1),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS}, RobPokerPlayer::PLAYER1)}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}, RobPokerPlayer::PLAYER2),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS}, RobPokerPlayer::PLAYER2)}},
            },

        .common_cards =
            {
                make_fog_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_3, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_6, Suits::SPADES}),
            },
        .actions = {SMALL_BLIND, BIG_BLIND},
    };
    for (const auto action : actions) {
        history = play(history, action);
    }

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
    {{CallAction{}}, RobPokerPlayer::PLAYER2},
    {{RaiseAction{}}, RobPokerPlayer::PLAYER2},
    {{FoldAction{}}, {}},
    {{CallAction{}, CheckAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, RaiseAction{}}, RobPokerPlayer::PLAYER1},
    {{CallAction{}, RaiseAction{}, RaiseAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}}, RobPokerPlayer::PLAYER1},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}},
     RobPokerPlayer::PLAYER1},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}},
     {}},
};
INSTANTIATE_TEST_SUITE_P(UpNext, RobPokerUpNextNoRunTest,
                         testing::ValuesIn(up_next_no_run_test_cases),
                         [](const auto &info) { return make_test_name_range_to_optional(info); });

class RobPokerUpNextWithRunTest : public RobPokerUpNextNoRunTest {};

TEST_P(RobPokerUpNextWithRunTest, test_up_next) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_3, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_6, Suits::HEARTS}),
                make_fog_card(SCard{Ranks::_7, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_8, Suits::CLUBS}),
            },
        .actions = {SMALL_BLIND, BIG_BLIND},
    };
    for (const auto action : actions) {
        history = play(history, action);
    }

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
    {{CallAction{}}, RobPokerPlayer::PLAYER2},
    {{RaiseAction{}}, RobPokerPlayer::PLAYER2},
    {{FoldAction{}}, {}},
    {{CallAction{}, CheckAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, RaiseAction{}}, RobPokerPlayer::PLAYER1},
    {{CallAction{}, RaiseAction{}, RaiseAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}}, RobPokerPlayer::PLAYER1},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}}, RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}},
     RobPokerPlayer::PLAYER1},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}},
     RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     RobPokerPlayer::PLAYER2},
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
};
INSTANTIATE_TEST_SUITE_P(UpNext, RobPokerUpNextWithRunTest,
                         testing::ValuesIn(up_next_with_run_test_cases),
                         [](const auto &info) { return make_test_name_range_to_optional(info); });

class RobPokerPossibleActionsTest
    : public testing::Test,
      public testing::WithParamInterface<std::pair<std::vector<Action>, std::vector<Action>>> {};

TEST_P(RobPokerPossibleActionsTest, test_possible_actions) {
    // Setup
    const auto &[action_history, expected_actions] = GetParam();
    RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_3, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_6, Suits::HEARTS}),
                make_fog_card(SCard{Ranks::_7, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_8, Suits::CLUBS}),
            },
        .actions = {SMALL_BLIND, BIG_BLIND},
    };

    for (const auto action : action_history) {
        history = play(history, action);
    }

    // Action
    const auto actions = possible_actions(history);

    // Verification
    EXPECT_EQ(expected_actions.size(), actions.size());
    for (const auto action : expected_actions) {
        const auto iter = std::find_if(actions.begin(), actions.end(), [&action](const auto &item) {
            return action.index() == item.index();
        });
        EXPECT_NE(iter, actions.end());
    }
}

std::pair<std::vector<Action>, std::vector<Action>> possible_actions_test_cases[] = {
    {{}, {CallAction{}, RaiseAction{}, FoldAction{}}},
    {{CallAction{}}, {RaiseAction{}, FoldAction{}, CheckAction{}}},
    {{RaiseAction{}}, {CallAction{}, RaiseAction{}, FoldAction{}}},
    {{FoldAction{}}, {}},
    {{CallAction{}, CheckAction{}}, {RaiseAction{}, CheckAction{}, FoldAction{}}},
    {{CallAction{}, RaiseAction{}}, {CallAction{}, RaiseAction{}, FoldAction{}}},
    {{CallAction{}, RaiseAction{}, RaiseAction{}}, {CallAction{}, RaiseAction{}, FoldAction{}}},
    // After first check in second betting round
    {{CallAction{}, CheckAction{}, CheckAction{}}, {RaiseAction{}, FoldAction{}, CheckAction{}}},
    // At start of third betting round
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {RaiseAction{}, FoldAction{}, CheckAction{}}},
    // After first check of fourth betting round
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}},
     {RaiseAction{}, FoldAction{}, CheckAction{}}},
    // At start of fifth betting round
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}},
     {RaiseAction{}, FoldAction{}, CheckAction{}}},
    // After first check of fifth betting round
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {RaiseAction{}, FoldAction{}, CheckAction{}}},
    // At the end of the last betting round
    {{CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
};
INSTANTIATE_TEST_SUITE_P(PossibleActions, RobPokerPossibleActionsTest,
                         testing::ValuesIn(possible_actions_test_cases),
                         [](const auto &info) { return make_test_name_range_to_range(info); });

TEST(RobPokerTest, test_game) {
    // Setup
    std::mt19937 gen(0);
    RobPokerHistory history;

    // Action
    auto maybe_current_player = up_next(history);
    while (maybe_current_player.has_value()) {
        const auto current_player = maybe_current_player.value();
        if (current_player == RobPokerPlayer::CHANCE) {
            const auto chance_result = play(history, make_in_out(gen));
            history = chance_result.history;
        } else {
            const auto actions = possible_actions(history);
            for (const auto desired_action : std::vector<Action>{CheckAction{}, CallAction{}}) {
                const auto iter = std::find_if(actions.begin(), actions.end(),
                                               [&desired_action](const auto &action) {
                                                   return desired_action.index() == action.index();
                                               });
                if (iter != actions.end()) {
                    history = play(history, desired_action);
                    break;
                }
            }
        }
        maybe_current_player = up_next(history);
    }

    // Verification
    const auto maybe_terminal_value = terminal_value(history, RobPokerPlayer::PLAYER1);
    EXPECT_TRUE(maybe_terminal_value.has_value());
}

TEST(RobPokerEvaluateHandTest, hand_evaluation) {
    // Setup
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{Ranks::_A, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_K, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_6, Suits::SPADES}),
            },
        .actions = {},
    };

    // Action
    const auto player_1_hand_rank = evaluate_hand(history, RobPokerPlayer::PLAYER1);
    const auto player_2_hand_rank = evaluate_hand(history, RobPokerPlayer::PLAYER2);

    // Verification
    EXPECT_GT(player_1_hand_rank, player_2_hand_rank);
}

class RobPokerTerminalValueNoRunTest
    : public testing::Test,
      public testing::WithParamInterface<std::pair<std::vector<Action>, std::optional<int>>> {};

TEST_P(RobPokerTerminalValueNoRunTest, test_terminal_value) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_public_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_3, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_7, Suits::SPADES}),
            },
        .actions = actions,
    };

    // Action
    const auto maybe_value = terminal_value(history, RobPokerPlayer::PLAYER1);

    // Verification
    ASSERT_EQ(maybe_value.has_value(), expected_value.has_value());
    if (expected_value.has_value()) {
        EXPECT_EQ(maybe_value.value(), expected_value.value());
    }
}

std::pair<std::vector<Action>, std::optional<int>> terminal_value_no_run_test_cases[] = {
    {{SMALL_BLIND, BIG_BLIND}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, RaiseAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, FoldAction{}}, -1},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, RaiseAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, RaiseAction{}, RaiseAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, FoldAction{}},
     2},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, FoldAction{}},
     -2},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     2},
};
INSTANTIATE_TEST_SUITE_P(TerminalValue, RobPokerTerminalValueNoRunTest,
                         testing::ValuesIn(terminal_value_no_run_test_cases), [](const auto &info) {
                             return make_test_name_range_to_optional_int(info);
                         });

class RobPokerTerminalValueWithRunTest : public RobPokerTerminalValueNoRunTest {};

TEST_P(RobPokerTerminalValueWithRunTest, test_terminal_value) {
    // Setup
    const auto &[actions, expected_value] = GetParam();
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_K, Suits::DIAMONDS}),
                  make_fog_card(SCard{Ranks::_K, Suits::HEARTS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_public_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_2, Suits::CLUBS}),
                make_public_card(SCard{Ranks::_2, Suits::SPADES}),
                make_public_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_6, Suits::HEARTS}),
                make_public_card(SCard{Ranks::_7, Suits::DIAMONDS}),
                make_public_card(SCard{Ranks::_8, Suits::CLUBS}),
            },
        .actions = actions,
    };

    // Action
    const auto maybe_value = terminal_value(history, RobPokerPlayer::PLAYER2);

    // Verification
    ASSERT_EQ(maybe_value.has_value(), expected_value.has_value());
    if (expected_value.has_value()) {
        EXPECT_EQ(maybe_value.value(), expected_value.value());
    }
}

std::pair<std::vector<Action>, std::optional<int>> terminal_value_with_run_test_cases[] = {
    {{SMALL_BLIND, BIG_BLIND}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, RAISE}, {}},
    {{SMALL_BLIND, BIG_BLIND, FoldAction{}}, 1},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, RAISE}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, RAISE, RAISE}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{}}, {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}},
     {}},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}},
     0},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, FoldAction{}},
     2},
    {{SMALL_BLIND, BIG_BLIND, CallAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{}, CheckAction{},
      FoldAction{}},
     -2},
};
INSTANTIATE_TEST_SUITE_P(TerminalValue, RobPokerTerminalValueWithRunTest,
                         testing::ValuesIn(terminal_value_with_run_test_cases),
                         [](const auto &info) {
                             return make_test_name_range_to_optional_int(info);
                         });

TEST(RobPokerTest, allin_call_is_game_end) {
    // Setup
    RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard{Ranks::_A, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_A, Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{Ranks::_K, Suits::SPADES}),
                  make_fog_card(SCard{Ranks::_K, Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{Ranks::_2, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_3, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_4, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_5, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_6, Suits::HEARTS}),
                make_fog_card(SCard{Ranks::_7, Suits::DIAMONDS}),
                make_fog_card(SCard{Ranks::_8, Suits::CLUBS}),
            },
        .actions = {SMALL_BLIND, BIG_BLIND, AllInAction{}},
    };

    // Action
    history = play(history, CallAction{});
    const auto maybe_value = terminal_value(history, RobPokerPlayer::PLAYER1);

    // Verification
    // While not part of the rules, we make the assumption that we must use at least one private
    // card
    ASSERT_TRUE(maybe_value.has_value());
    EXPECT_EQ(maybe_value.value(), 400);
}

TEST(ComputeBettingStateTest, five_card_board_has_correct_betting_state) {
    // Setup
    const RobPokerHistory history = {
        .hole_cards =
            {
                {RobPokerPlayer::PLAYER1,
                 {make_fog_card(SCard(Ranks::_5, Suits::HEARTS), RobPokerPlayer::PLAYER1),
                  make_fog_card(SCard(Ranks::_K, Suits::DIAMONDS), RobPokerPlayer::PLAYER1)}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard(Ranks::_7, Suits::CLUBS), RobPokerPlayer::PLAYER2),
                  make_fog_card(SCard(Ranks::_3, Suits::CLUBS), RobPokerPlayer::PLAYER2)}},
            },
        .common_cards =
            {
                make_public_card(SCard(Ranks::_7, Suits::HEARTS)),
                make_public_card(SCard(Ranks::_9, Suits::HEARTS)),
                make_public_card(SCard(Ranks::_2, Suits::HEARTS)),
                make_public_card(SCard(Ranks::_J, Suits::CLUBS)),
                make_public_card(SCard(Ranks::_6, Suits::SPADES)),
            },
        .actions = {SMALL_BLIND, BIG_BLIND, CallAction(), CheckAction(), CheckAction(),
                    CheckAction(), CheckAction(), CheckAction()}};

    // Action
    const auto betting_state = compute_betting_state(history);

    // Verification
    EXPECT_FALSE(betting_state.is_game_over);
    EXPECT_FALSE(betting_state.showdown_required);
    EXPECT_TRUE(betting_state.to_bet.has_value());
    EXPECT_EQ(betting_state.to_bet->round, 3);
    EXPECT_EQ(betting_state.to_bet->position, 0);
    EXPECT_TRUE(betting_state.to_bet->is_final_betting_round);
}

}  // namespace robot::domain
