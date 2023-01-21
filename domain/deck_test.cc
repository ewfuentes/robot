
#include "deck.hh"

#include <unordered_set>

#include "gtest/gtest.h"

#include "absl/container/flat_hash_map.h"

namespace robot::domain {
TEST(DeckTest, deck_contains_all_cards) {
    // Setup
    StandardDeck deck;

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

TEST(DeckTest, deal_all_combinations_test) {
    // Setup
    std::mt19937 gen(0);
    StandardDeck deck;

    // Action
    absl::flat_hash_map<std::pair<StandardDeck::Card, StandardDeck::Card>, int> counts_map;
    for (int i = 0; i < 100000; i++) {
        StandardDeck deck;
        deck.shuffle(make_in_out(gen));
        const auto card_1 = deck.deal_card().value();
        const auto card_2 = deck.deal_card().value();
        counts_map[{card_1, card_2}]++;
    }

    // Verification
    EXPECT_EQ(counts_map.size(), 52 * 51);

}
}  // namespace robot::domain
