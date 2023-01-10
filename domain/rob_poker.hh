
#pragma once

#include <random>

#include "common/argument_wrapper.hh"
#include "common/indexed_array.hh"
#include "domain/deck.hh"
#include "domain/fog.hh"
#include "wise_enum.h"

namespace robot::domain {
WISE_ENUM_CLASS(RobPokerAction, FOLD, CHECK, CALL, RAISE);
WISE_ENUM_CLASS(RobPokerPlayer, PLAYER1, PLAYER2, CHANCE);

std::ostream &operator<<(std::ostream &out, const RobPokerPlayer player);
std::ostream &operator<<(std::ostream &out, const RobPokerAction player);

struct RobPokerHistory {
    // TODO Fog card really seems like overkill... We could replace the function call with a boolean
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
    using InfoSetId = std::string;
};

ChanceResult play(const RobPokerHistory &history, InOut<std::mt19937> gen);
RobPokerHistory play(const RobPokerHistory &history, const RobPokerAction &action);
std::optional<RobPokerPlayer> up_next(const RobPokerHistory &history);
std::vector<RobPokerAction> possible_actions(const RobPokerHistory &history);
std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player);

RobPoker::InfoSetId infoset_id_from_history(const RobPokerHistory &hist);
RobPoker::InfoSetId infoset_id_from_information(
    const std::array<StandardDeck::Card, 2> &private_cards,
    const std::vector<RobPokerAction> &actions);
std::string to_string(const RobPokerHistory &hist);

}  // namespace robot::domain
