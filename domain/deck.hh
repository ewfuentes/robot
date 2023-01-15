
#pragma once

#include <array>
#include <cctype>
#include <optional>
#include <random>
#include <sstream>

#include "common/argument_wrapper.hh"
#include "omp/Random.h"
#include "wise_enum.h"

namespace robot::domain {

WISE_ENUM_CLASS(StandardSuits, SPADES, HEARTS, CLUBS, DIAMONDS)
WISE_ENUM_CLASS(StandardRanks, _2, _3, _4, _5, _6, _7, _8, _9, _T, _J, _Q, _K, _A)

template <typename RankEnum, typename SuitEnum>
struct CardT {
    using Ranks = RankEnum;
    using Suits = SuitEnum;

    RankEnum rank;
    SuitEnum suit;

    bool operator==(const CardT &other) const { return rank == other.rank && suit == other.suit; }
};

template <typename T>
concept CardLike = requires(T a) {
    a.rank;
    a.suit;
};

template <CardLike T>
struct CardHash {
    std::size_t operator()(const T &card) const {
        return (static_cast<int>(card.rank) << 16) | static_cast<int>(card.suit);
    }
};

template <CardLike Card>
std::string to_string(const Card &card) {
    std::stringstream out;
    out << wise_enum::to_string(card.rank)[1]
        << static_cast<char>(std::tolower(wise_enum::to_string(card.suit)[0]));
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
        bottom_location_ = NUM_CARDS;
    }

    void shuffle(InOut<std::mt19937> gen) {
        // seed a faster random number generator using the generator passed in
        omp::XoroShiro128Plus fast_gen((*gen)());
        for (int i = bottom_location_ - 1; i > top_location_; i--) {
            // Work from the end to the beginning, picking which card goes in the i'th position
            omp::FastUniformIntDistribution<> card_picker(top_location_, i);
            const int idx = card_picker(fast_gen);
            // Swap the cards at the i'th and the idx'th position
            std::swap(elements_[i], elements_[idx]);
        }
    }

    std::optional<Card> deal_card() {
        if (top_location_ < bottom_location_) {
            return elements_[top_location_++];
        }
        return std::nullopt;
    }

    int size() const { return bottom_location_ - top_location_; }

    constexpr typename Container::iterator erase(const typename Container::iterator &new_end) {
        bottom_location_ = std::distance(begin(), new_end);
        return end();
    }

    constexpr typename Container::const_iterator begin() const {
        return elements_.begin() + top_location_;
    };
    constexpr typename Container::const_iterator end() const {
        return elements_.begin() + bottom_location_;
    };
    constexpr typename Container::iterator begin() { return elements_.begin() + top_location_; };
    constexpr typename Container::iterator end() { return elements_.begin() + bottom_location_; };

   private:
    Container elements_;
    int top_location_;
    int bottom_location_;
};

using StandardDeck = Deck<StandardRanks, StandardSuits>;

}  // namespace robot::domain
