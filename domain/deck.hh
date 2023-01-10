
#pragma once

#include <array>
#include <optional>
#include <random>
#include <sstream>

#include "common/argument_wrapper.hh"
#include "wise_enum.h"

namespace robot::domain {

WISE_ENUM_CLASS(StandardSuits, SPADES, HEARTS, CLUBS, DIAMONDS);
WISE_ENUM_CLASS(StandardRanks, _2, _3, _4, _5, _6, _7, _8, _9, _10, _J, _Q, _K, _A);

template <typename RankEnum, typename SuitEnum>
struct CardT {
    using Ranks = RankEnum;
    using Suits = SuitEnum;

    RankEnum rank;
    SuitEnum suit;

    bool operator==(const CardT &other) const { return rank == other.rank && suit == other.suit; }
};

template <typename T>
struct CardHash {
    std::size_t operator()(const T &card) const {
        return (static_cast<int>(card.rank) << 16) | static_cast<int>(card.suit);
    }
};

template <typename CardT>
std::string to_string(const CardT &card) {
    std::stringstream out;
    out << wise_enum::to_string(card.rank) << wise_enum::to_string(card.suit);
    return out.str();
}

template <typename RankEnum, typename SuitEnum>
struct Deck {
   public:
    using Ranks = RankEnum;
    using Suits = SuitEnum;
    using Card = CardT<Ranks, Suits>;
    static constexpr int RANK_SIZE = wise_enum::size<Ranks>;
    static constexpr int SUIT_SIZE = wise_enum::size<Suits>;
    static constexpr int NUM_CARDS = RANK_SIZE * SUIT_SIZE;
    using Container = std::array<Card, NUM_CARDS>;

    constexpr Deck() {
        int i = 0;
        for (const auto &[rank, _] : wise_enum::range<Ranks>) {
            for (const auto &[suit, _] : wise_enum::range<Suits>) {
                elements_[i++] = {rank, suit};
            }
        }
        top_location_ = 0;
    }

    void shuffle(InOut<std::mt19937> gen) {
        for (int i = NUM_CARDS - 1; i > top_location_; i--) {
            // Work from the end to the beginning, picking which card goes in the i'th position
            std::uniform_int_distribution<> card_picker(top_location_, i);
            const int idx = card_picker(*gen);
            // Swap the cards at the i'th and the idx'th position
            std::swap(elements_[i], elements_[idx]);
        }
    }

    std::optional<Card> deal_card() {
        if (top_location_ < NUM_CARDS) {
            return elements_[top_location_++];
        }
        return std::nullopt;
    }

    int size() const { return NUM_CARDS - top_location_; }

    constexpr typename Container::const_iterator begin() const {
        return elements_.begin() + top_location_;
    };
    constexpr typename Container::const_iterator end() const { return elements_.end(); };
    constexpr typename Container::iterator begin() { return elements_.begin() + top_location_; };
    constexpr typename Container::iterator end() { return elements_.end(); };

   private:
    Container elements_;
    int top_location_;
};

using StandardDeck = Deck<StandardRanks, StandardSuits>;

}  // namespace robot::domain
