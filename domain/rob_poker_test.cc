
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

std::string range_to_string(const auto &range) {
    std::stringstream out;
    out << (range.empty() ? "none" : "");
    for (const auto &item : range) {
        out << wise_enum::to_string(item);
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
                         [](const auto &info) { return make_test_name_range_to_optional(info); });

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
                         [](const auto &info) { return make_test_name_range_to_optional(info); });

class RobPokerPossibleActionsTest
    : public testing::Test,
      public testing::WithParamInterface<std::pair<std::vector<Action>, std::vector<Action>>> {};

TEST_P(RobPokerPossibleActionsTest, test_possible_actions) {
    // Setup
    const auto &[action_history, expected_actions] = GetParam();
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
        .actions = action_history,
    };

    // Action
    const auto actions = possible_actions(history);

    // Verification
    EXPECT_EQ(expected_actions.size(), actions.size());
    for (const auto action : expected_actions) {
        const auto iter = std::find(actions.begin(), actions.end(), action);
        EXPECT_NE(iter, actions.end());
    }
}

std::pair<std::vector<Action>, std::vector<Action>> possible_actions_test_cases[] = {
    {{}, {Action::CALL, Action::RAISE, Action::FOLD}},
    {{Action::CALL}, {Action::RAISE, Action::FOLD, Action::CHECK}},
    {{Action::RAISE}, {Action::CALL, Action::RAISE, Action::FOLD}},
    {{Action::FOLD}, {}},
    {{Action::CALL, Action::CHECK}, {Action::RAISE, Action::CHECK, Action::FOLD}},
    {{Action::CALL, Action::RAISE}, {Action::CALL, Action::RAISE, Action::FOLD}},
    {{Action::CALL, Action::RAISE, Action::RAISE}, {Action::CALL, Action::RAISE, Action::FOLD}},
    // After first check in second betting round
    {{Action::CALL, Action::CHECK, Action::CHECK}, {Action::RAISE, Action::FOLD, Action::CHECK}},
    // At start of third betting round
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK},
     {Action::RAISE, Action::FOLD, Action::CHECK}},
    // After first check of fourth betting round
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK},
     {Action::RAISE, Action::FOLD, Action::CHECK}},
    // At start of fifth betting round
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK},
     {Action::RAISE, Action::FOLD, Action::CHECK}},
    // After first check of fifth betting round
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
     {Action::RAISE, Action::FOLD, Action::CHECK}},
    // At the end of the last betting round
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
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
            for (const auto desired_action : {RobPokerAction::CHECK, RobPokerAction::CALL}) {
                const auto iter = std::find(actions.begin(), actions.end(), desired_action);
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
                 {make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::CLUBS})}},
                {RobPokerPlayer::PLAYER2,
                 {make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::SPADES}),
                  make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::CLUBS})}},
            },

        .common_cards =
            {
                make_fog_card(SCard{.rank = Ranks::_A, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_K, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_4, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_5, .suit = Suits::DIAMONDS}),
                make_fog_card(SCard{.rank = Ranks::_6, .suit = Suits::SPADES}),
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
                make_fog_card(SCard{.rank = Ranks::_7, .suit = Suits::SPADES}),
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
    {{}, {}},
    {{Action::CALL}, {}},
    {{Action::RAISE}, {}},
    {{Action::FOLD}, -1},
    {{Action::CALL, Action::CHECK}, {}},
    {{Action::CALL, Action::RAISE}, {}},
    {{Action::CALL, Action::RAISE, Action::RAISE}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK},
     {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::FOLD},
     1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::FOLD},
     -1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK},
     1},
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
    const auto maybe_value = terminal_value(history, RobPokerPlayer::PLAYER2);

    // Verification
    ASSERT_EQ(maybe_value.has_value(), expected_value.has_value());
    if (expected_value.has_value()) {
        EXPECT_EQ(maybe_value.value(), expected_value.value());
    }
}

std::pair<std::vector<Action>, std::optional<int>> terminal_value_with_run_test_cases[] = {
    {{}, {}},
    {{Action::CALL}, {}},
    {{Action::RAISE}, {}},
    {{Action::FOLD}, 1},
    {{Action::CALL, Action::CHECK}, {}},
    {{Action::CALL, Action::RAISE}, {}},
    {{Action::CALL, Action::RAISE, Action::RAISE}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK}, {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK},
     {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK},
     {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
     {}},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK},
     0},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::FOLD},
     1},
    {{Action::CALL, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK,
      Action::CHECK, Action::CHECK, Action::CHECK, Action::CHECK, Action::FOLD},
     -1},
};
INSTANTIATE_TEST_SUITE_P(TerminalValue, RobPokerTerminalValueWithRunTest,
                         testing::ValuesIn(terminal_value_with_run_test_cases),
                         [](const auto &info) {
                             return make_test_name_range_to_optional_int(info);
                         });

}  // namespace robot::domain
