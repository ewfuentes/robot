
#include <optional>
#include <string>

#include "domain/fog.hh"
#include "wise_enum.h"

namespace robot::domain {

WISE_ENUM_CLASS(RPSAction, ROCK, PAPER, SCISSORS);
WISE_ENUM_CLASS(RPSPlayer, PLAYER1, PLAYER2);

struct RPSHistory {
    using FogAction = Fog<RPSAction, RPSPlayer>;
    FogAction player_1_action;
    FogAction player_2_action;
};

struct RockPaperScissors {
    using Players = RPSPlayer;
    using Actions = RPSAction;
    using History = RPSHistory;
    using InfoSetId = std::string;
};

RPSHistory play(const RPSHistory &history, const RPSAction &action);

std::optional<RPSPlayer> up_next(const RPSHistory &history);

std::vector<RPSAction> possible_actions(const RPSHistory &history);

std::optional<int> terminal_value(const RPSHistory &history, const RPSPlayer player);
}  // namespace robot::domain
