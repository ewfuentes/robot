
#include "domain/rock_paper_scissors.hh"

#include "gtest/gtest.h"

namespace robot::domain {
TEST(RPSTest, empty_history) {
    // Setup
    const RPSHistory history;

    // Action + Verification
    EXPECT_EQ(up_next(history), RPSPlayer::PLAYER1);
    EXPECT_EQ(terminal_value(history, RPSPlayer::PLAYER1), std::nullopt);
    EXPECT_EQ(terminal_value(history, RPSPlayer::PLAYER2), std::nullopt);
}

TEST(RPSTest, player_1_wins) {
    // Setup
    RPSHistory history;

    // Action
    history = play(history, RPSAction::ROCK);
    history = play(history, RPSAction::SCISSORS);

    // Verification
    EXPECT_EQ(up_next(history), std::nullopt);

    const auto p1_terminal_value = terminal_value(history, RPSPlayer::PLAYER1);
    EXPECT_TRUE(p1_terminal_value.has_value());
    EXPECT_EQ(p1_terminal_value.value(), 1);

    const auto p2_terminal_value = terminal_value(history, RPSPlayer::PLAYER2);
    EXPECT_TRUE(p2_terminal_value.has_value());
    EXPECT_EQ(p2_terminal_value.value(), -1);
}

TEST(RPSTest, player_2_wins) {
    // Setup
    RPSHistory history;

    // Action
    history = play(history, RPSAction::PAPER);
    history = play(history, RPSAction::SCISSORS);

    // Verification
    EXPECT_EQ(up_next(history), std::nullopt);

    const auto p1_terminal_value = terminal_value(history, RPSPlayer::PLAYER1);
    EXPECT_TRUE(p1_terminal_value.has_value());
    EXPECT_EQ(p1_terminal_value.value(), -1);

    const auto p2_terminal_value = terminal_value(history, RPSPlayer::PLAYER2);
    EXPECT_TRUE(p2_terminal_value.has_value());
    EXPECT_EQ(p2_terminal_value.value(), 1);
}

TEST(RPSTest, tie_game) {
    // Setup
    RPSHistory history;

    // Action
    history = play(history, RPSAction::ROCK);
    history = play(history, RPSAction::ROCK);

    // Verification
    EXPECT_EQ(up_next(history), std::nullopt);

    const auto p1_terminal_value = terminal_value(history, RPSPlayer::PLAYER1);
    EXPECT_TRUE(p1_terminal_value.has_value());
    EXPECT_EQ(p1_terminal_value.value(), 0);

    const auto p2_terminal_value = terminal_value(history, RPSPlayer::PLAYER2);
    EXPECT_TRUE(p2_terminal_value.has_value());
    EXPECT_EQ(p2_terminal_value.value(), 0);
}
}  // namespace robot::domain
