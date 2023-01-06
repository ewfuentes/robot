
#include "domain/kuhn_poker.hh"

#include "gtest/gtest.h"

namespace robot::domain {
TEST(KuhnPokerTest, deal_required_before_actions) {
    // Setup
    KuhnHistory history;
    std::mt19937 gen(0);

    // Action + Verification
    {
        // Check that chance is the required player
        const auto maybe_next_player = up_next(history);
        EXPECT_TRUE(maybe_next_player.has_value());
        EXPECT_EQ(maybe_next_player.value(), KuhnPlayer::CHANCE);
    }
    {
        // Check that attempting to play a bet or action leads to the game not advancing
        for (const auto action : {KuhnAction::BET, KuhnAction::PASS}) {
            history = play(history, action);
            const auto maybe_next_player = up_next(history);
            EXPECT_TRUE(maybe_next_player.has_value());
            EXPECT_EQ(maybe_next_player.value(), KuhnPlayer::CHANCE);
        }
    }
    {
        // Check that dealing leads to the game advancing
        history = play(history, make_in_out(gen));
        const auto maybe_next_player = up_next(history);
        EXPECT_TRUE(maybe_next_player.has_value());
        EXPECT_NE(maybe_next_player.value(), KuhnPlayer::CHANCE);
    }
}

TEST(KuhnPokerTest, passes_lead_to_highest_card_winning) {
    // Setup
    using Card = KuhnHistory::Card;
    using P = KuhnPlayer;
    const KuhnHistory history = {.hands = {{P::PLAYER1, {Card::A, make_private_info(P::PLAYER1)}},
                                           {P::PLAYER2, {Card::K, make_private_info(P::PLAYER2)}}},
                                 .actions = {KuhnAction::PASS, KuhnAction::PASS}};
    constexpr P EXPECTED_WINNER = P::PLAYER1;
    constexpr int EXPECTED_VALUE = 1;

    // Action + Verification
    for (const auto player : {P::PLAYER1, P ::PLAYER2}) {
        const auto maybe_value = terminal_value(history, player);
        EXPECT_TRUE(maybe_value.has_value());
        const int sign = player == EXPECTED_WINNER ? 1 : -1;
        EXPECT_EQ(maybe_value.value(), sign * EXPECTED_VALUE);
    }
}

TEST(KuhnPokerTest, bets_lead_to_highest_card_winning) {
    // Setup
    using Card = KuhnHistory::Card;
    using P = KuhnPlayer;
    const KuhnHistory history = {.hands = {{P::PLAYER1, {Card::A, make_private_info(P::PLAYER1)}},
                                           {P::PLAYER2, {Card::K, make_private_info(P::PLAYER2)}}},
                                 .actions = {KuhnAction::BET, KuhnAction::BET}};
    constexpr P EXPECTED_WINNER = P::PLAYER1;
    constexpr int EXPECTED_VALUE = 2;

    // Action + Verification
    for (const auto player : {P::PLAYER1, P ::PLAYER2}) {
        const auto maybe_value = terminal_value(history, player);
        EXPECT_TRUE(maybe_value.has_value());
        const int sign = player == EXPECTED_WINNER ? 1 : -1;
        EXPECT_EQ(maybe_value.value(), sign * EXPECTED_VALUE);
    }
}

TEST(KuhnPokerTest, bets_after_player_1_pass_lead_to_highest_card_winning) {
    // Setup
    using Card = KuhnHistory::Card;
    using P = KuhnPlayer;
    const KuhnHistory history = {.hands = {{P::PLAYER1, {Card::Q, make_private_info(P::PLAYER1)}},
                                           {P::PLAYER2, {Card::K, make_private_info(P::PLAYER2)}}},
                                 .actions = {KuhnAction::PASS, KuhnAction::BET, KuhnAction::BET}};
    constexpr P EXPECTED_WINNER = P::PLAYER2;
    constexpr int EXPECTED_VALUE = 2;

    // Action + Verification
    for (const auto player : {P::PLAYER1, P ::PLAYER2}) {
        const auto maybe_value = terminal_value(history, player);
        EXPECT_TRUE(maybe_value.has_value());
        const int sign = player == EXPECTED_WINNER ? 1 : -1;
        EXPECT_EQ(maybe_value.value(), sign * EXPECTED_VALUE);
    }
}

TEST(KuhnPokerTest, player_1_pass_after_bet_ends_game) {
    // Setup
    using Card = KuhnHistory::Card;
    using P = KuhnPlayer;
    const KuhnHistory history = {.hands = {{P::PLAYER1, {Card::A, make_private_info(P::PLAYER1)}},
                                           {P::PLAYER2, {Card::K, make_private_info(P::PLAYER2)}}},
                                 .actions = {KuhnAction::PASS, KuhnAction::BET, KuhnAction::PASS}};
    constexpr P EXPECTED_WINNER = P::PLAYER2;
    constexpr int EXPECTED_VALUE = 1;

    // Action + Verification
    for (const auto player : {P::PLAYER1, P ::PLAYER2}) {
        const auto maybe_value = terminal_value(history, player);
        EXPECT_TRUE(maybe_value.has_value());
        const int sign = player == EXPECTED_WINNER ? 1 : -1;
        EXPECT_EQ(maybe_value.value(), sign * EXPECTED_VALUE);
    }
}

TEST(KuhnPokerTest, player_2_pass_after_bet_ends_game) {
    // Setup
    using Card = KuhnHistory::Card;
    using P = KuhnPlayer;
    const KuhnHistory history = {.hands = {{P::PLAYER1, {Card::A, make_private_info(P::PLAYER1)}},
                                           {P::PLAYER2, {Card::K, make_private_info(P::PLAYER2)}}},
                                 .actions = {KuhnAction::BET, KuhnAction::PASS}};
    constexpr P EXPECTED_WINNER = P::PLAYER1;
    constexpr int EXPECTED_VALUE = 1;

    // Action + Verification
    for (const auto player : {P::PLAYER1, P ::PLAYER2}) {
        const auto maybe_value = terminal_value(history, player);
        EXPECT_TRUE(maybe_value.has_value());
        const int sign = player == EXPECTED_WINNER ? 1 : -1;
        EXPECT_EQ(maybe_value.value(), sign * EXPECTED_VALUE);
    }
}
}  // namespace robot::domain
