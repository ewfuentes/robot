
#include <optional>

#include "domain/fog.hh"

namespace robot::domain {

enum class RPSAction { ROCK, PAPER, SCISSORS };
enum class RPSPlayer { PLAYER1, PLAYER2 };

struct RPSHistory {
    using FogAction = Fog<RPSAction, RPSPlayer>;
    FogAction player_1_action;
    FogAction player_2_action;
};

RPSHistory play(const RPSHistory &history, const RPSAction &action);

std::optional<RPSPlayer> up_next(const RPSHistory &history);

std::vector<RPSAction> possible_actions(const RPSHistory &history);

std::optional<int> terminal_value(const RPSHistory &history, const RPSPlayer player);
}  // namespace robot::domain
