
#pragma once

#include "wise_enum.h"

namespace robot::domain {
WISE_ENUM_CLASS(RobPokerAction, CHECK, CALL, BET);
WISE_ENUM_CLASS(RobPlayer, PLAYER1, PLAYER2, CHANCE);

struct RobPokerHistory {
    using FogCard = Fog<StandardDeck::Card, RobPlayer>;
    IndexedArray<std::array<StandardDeck::Card, 2>, RobPlayer> hole_cards;
    std::array<FogCard<StandardDeck::Card>, 20> common_cards;
};

struct ChanceResult {
  RobPokerHistory history;
  double probability;
};


struct RobPoker {
    using Players = RobPlayer;
    using Actions = RobPokerAction;
    using History = RobPokerHistory;
    using InfoSetId = std::string;
};

RobPokerHistory play(const RobPokerHistory &history, const RobPokerAction &action);
ChanceResult play(const RobPokerHistory &history, InOut<std::mt19937> gen);
std::optional<RobPokerPlayer> up_next(const RobPokerHistory &history);
std::vector<RobPokerAction> possible_actions(const RobPokerHistory &history);
std::optional<int> terminal_value(const RobPokerHistory &history, const RobPokerPlayer player);

RobPokerPoker::InfoSetId infoset_id_from_history(const RobPokerHistory &hist);
RobPokerPoker::InfoSetId infoset_id_from_information(const RobPokerHistory::Card private_card,
                                                 const std::vector<RobPokerAction> &actions);
std::string to_string(const RobPokerHistory &hist);

}  // namespace robot::domain
