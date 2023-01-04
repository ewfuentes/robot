
#include "domain/rock_paper_scissors.hh"

namespace robot::domain {

std::optional<RPSPlayer> up_next(const RPSHistory &history) {
    if (!history.player_1_action.has_value()) {
        return RPSPlayer::PLAYER1;
    } else if (!history.player_2_action.has_value()) {
        return RPSPlayer::PLAYER2;
    }
    return std::nullopt;
}

std::optional<int> terminal_value(const RPSHistory &history, const RPSPlayer player) {
    if (up_next(history).has_value()) {
        return std::nullopt;
    }
    const RPSAction &current_action = player == RPSPlayer::PLAYER1
                                          ? history.player_1_action.value()
                                          : history.player_2_action.value();
    const RPSAction &other_action = player == RPSPlayer::PLAYER1 ? history.player_2_action.value()
                                                                 : history.player_1_action.value();

    if (current_action == other_action) {
        return 0;
    } else if (current_action == RPSAction::ROCK) {
        if (other_action == RPSAction::PAPER) {
            return -1;
        }
        return 1;
    } else if (current_action == RPSAction::PAPER) {
        if (other_action == RPSAction::SCISSORS) {
            return -1;
        }
        return 1;
    } else if (current_action == RPSAction::SCISSORS) {
        if (other_action == RPSAction::ROCK) {
            return -1;
        }
        return 1;
    }

    return std::nullopt;
}

RPSHistory play(const RPSHistory &history, const RPSAction &action) {
    const auto maybe_next_player = up_next(history);
    if (!maybe_next_player.has_value()) {
        return history;
    }

    const auto visible_to_func = [visible_player = maybe_next_player.value()](const auto player) {
        return visible_player == player;
    };

    RPSHistory out = history;
    if (maybe_next_player.value() == RPSPlayer::PLAYER1) {
        out.player_1_action = RPSHistory::FogAction(action, visible_to_func);
    } else {
        out.player_2_action = RPSHistory::FogAction(action, visible_to_func);
    }
    return out;
}

double compute_counterfactual_regret(const RPSHistory &history, const RPSPlayer player,
                                     const RPSAction new_action) {
    RPSHistory counterfactual = history;
    RPSHistory::FogAction &player_action = player == RPSPlayer::PLAYER1
                                               ? counterfactual.player_1_action
                                               : counterfactual.player_2_action;
    player_action = RPSHistory::FogAction(new_action, [](...) { return true; });
    return terminal_value(counterfactual, player).value() - terminal_value(history, player).value();
}
}  // namespace robot::domain
