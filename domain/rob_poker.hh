
#pragma once

#include <random>
#include <span>
#include <variant>

#include "common/argument_wrapper.hh"
#include "common/indexed_array.hh"
#include "domain/deck.hh"
#include "domain/fog.hh"
#include "wise_enum.h"

namespace robot::domain {
struct FoldAction {
    static constexpr std::string_view name = "Fold";
    constexpr bool operator==(const FoldAction &) const { return true; }
};
struct CheckAction {
    static constexpr std::string_view name = "Check";
    constexpr bool operator==(const CheckAction &) const { return true; }
};
struct CallAction {
    static constexpr std::string_view name = "Call";
    constexpr bool operator==(const CallAction &) const { return true; }
};
struct RaiseAction {
    static constexpr std::string_view name = "Raise";
    int amount;
    constexpr bool operator==(const RaiseAction &other) const { return amount == other.amount; }
};

struct RaisePotAction {
    static constexpr std::string_view name = "RaisePot";
    constexpr bool operator==(const RaisePotAction &) const { return true; }
};

struct AllInAction {
    static constexpr std::string_view name = "AllIn";
    constexpr bool operator==(const AllInAction &) const { return true; }
};

using RobPokerAction =
    std::variant<FoldAction, CheckAction, CallAction, RaiseAction, RaisePotAction, AllInAction>;
WISE_ENUM_CLASS(RobPokerPlayer, PLAYER1, PLAYER2, CHANCE)

struct BettingState {
    bool is_game_over;
    bool showdown_required;
    bool is_final_betting_round;
    int round;
    int position;
    IndexedArray<int, RobPokerPlayer> put_in_pot;
};

struct RobPokerHistory {
    static constexpr int STARTING_STACK_SIZE = 400;
    static constexpr int SMALL_BLIND = 1;
    static constexpr int BIG_BLIND = 2;

    // TODO Fog card really seems like overkill... We could replace the function call with a
    // boolean
    using FogCard = Fog<StandardDeck::Card, RobPokerPlayer>;
    IndexedArray<std::array<FogCard, 2>, RobPokerPlayer> hole_cards;
    // 26 red cards + 5 black cards is the longest it could be
    std::array<FogCard, 31> common_cards;

    std::vector<RobPokerAction> actions;
};

struct ChanceResult {
    RobPokerHistory history;
    double probability;
};

struct RobPoker {
    using Players = RobPokerPlayer;
    using Actions = RobPokerAction;
    using History = RobPokerHistory;
    using InfoSetId = uint64_t;
};

std::ostream &operator<<(std::ostream &out, const RobPokerHistory &history);
std::ostream &operator<<(std::ostream &out, const RobPokerPlayer player);
std::ostream &operator<<(std::ostream &out, const RobPokerAction action);

ChanceResult play(const RobPokerHistory &history, InOut<std::mt19937> gen);
RobPokerHistory play(const RobPokerHistory &history, const RobPokerAction &action);
std::optional<RobPokerPlayer> up_next(const RobPokerHistory &history);
std::vector<RobPokerAction> possible_actions(const RobPokerHistory &history);
std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player);

std::string to_string(const RobPokerHistory &hist);
std::string to_string(const RobPokerAction &hist);

int evaluate_hand(const RobPokerHistory &history, const RobPokerPlayer player);
int evaluate_hand(const std::array<StandardDeck::Card, 33> &cards, const int num_cards);
BettingState compute_betting_state(const RobPokerHistory &history);
namespace detail {
  std::span<const std::vector<int>> n_choose_k(const int n, const int k);
}
}  // namespace robot::domain
