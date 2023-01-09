
#include "deck.hh"

#include <unordered_set>

#include "gtest/gtest.h"

namespace robot::domain {
TEST(DeckTest, deck_contains_all_cards) {
    // Setup
    constexpr StandardDeck deck;

    // Action + Verification
    EXPECT_EQ(deck.size(), StandardDeck::RANK_SIZE * StandardDeck::SUIT_SIZE);
    std::unordered_set<StandardDeck::Card, CardHash<StandardDeck::Card>> card_set;

    for (const auto card : deck) {
        card_set.insert(card);
    }

    EXPECT_EQ(card_set.size(), StandardDeck::RANK_SIZE * StandardDeck::SUIT_SIZE);
}

TEST(DeckTest, deck_contains_all_cards_after_shuffle) {
    // Setup
    std::mt19937 gen(0);
    StandardDeck deck;

    // Action
    deck.shuffle(make_in_out(gen));

    // Verification
    EXPECT_EQ(deck.size(), StandardDeck::RANK_SIZE * StandardDeck::SUIT_SIZE);
    std::unordered_set<StandardDeck::Card, CardHash<StandardDeck::Card>> card_set;

    for (const auto card : deck) {
        card_set.insert(card);
    }

    EXPECT_EQ(card_set.size(), StandardDeck::RANK_SIZE * StandardDeck::SUIT_SIZE);
}

TEST(DeckTest, dealing_reduces_size) {
    // Setup
    constexpr int NUM_TO_DEAL = 10;
    std::mt19937 gen(0);
    StandardDeck deck;

    // Action
    deck.shuffle(make_in_out(gen));
    for (int i = 0; i < 10; i++) {
        deck.deal_card();
    }

    // Verification
    EXPECT_EQ(deck.size(), StandardDeck::NUM_CARDS - NUM_TO_DEAL);
}
}  // namespace robot::domain
