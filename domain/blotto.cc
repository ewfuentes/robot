
#include "domain/blotto.hh"

namespace robot::domain {
BlottoHistory play(const BlottoHistory &history, const BlottoAction &action) {
    auto out = history;
    if (!history.player_1_action.has_value()) {
        out.player_1_action = BlottoHistory::FogAction(
            action, [](const auto player) { return player == BlottoPlayer::PLAYER1; });
    } else if (!history.player_2_action.has_value()) {
        out.player_2_action = BlottoHistory::FogAction(
            action, [](const auto player) { return player == BlottoPlayer::PLAYER2; });
    }
    return out;
}

std::optional<BlottoPlayer> up_next(const BlottoHistory &history) {
    if (!history.player_1_action.has_value()) {
        return BlottoPlayer::PLAYER1;
    } else if (!history.player_2_action.has_value()) {
        return BlottoPlayer::PLAYER2;
    }
    return std::nullopt;
}

std::vector<BlottoAction> possible_actions(const BlottoHistory &history) {
    if (!up_next(history).has_value()) {
        return {};
    }

    std::vector<BlottoAction> actions;
    std::transform(wise_enum::range<BlottoAction>.begin(), wise_enum::range<BlottoAction>.end(),
                   std::back_inserter(actions),
                   [](const auto &action_and_name) { return action_and_name.value; });
    return actions;
}

std::optional<int> terminal_value(const BlottoHistory &history, const BlottoPlayer player) {
    if (up_next(history).has_value()) {
        return std::nullopt;
    }
    const BlottoAction curr_action = player == BlottoPlayer::PLAYER1
                                         ? history.player_1_action.value()
                                         : history.player_2_action.value();
    const BlottoAction opp_action = player == BlottoPlayer::PLAYER1
                                        ? history.player_2_action.value()
                                        : history.player_1_action.value();

    auto assignments_from_action = [](const auto action) {
        const int value = static_cast<int>(action);
        return std::vector<int>{value % 10, (value / 10) % 10, (value / 100) % 10};
    };
    const auto curr_value = assignments_from_action(curr_action);
    const auto opp_value = assignments_from_action(opp_action);

    int terminal_value = 0;
    for (int i = 0; i < static_cast<int>(curr_value.size()); i++) {
        if (curr_value.at(i) > opp_value.at(i)) {
            terminal_value++;
        } else if (curr_value.at(i) < opp_value.at(i)) {
            terminal_value--;
        }
    }
    return terminal_value;
}

double compute_counterfactual_regret(const BlottoHistory &history, const BlottoPlayer player,
                                     const BlottoAction new_action) {
    BlottoHistory counterfactual = history;
    auto &player_action = player == BlottoPlayer::PLAYER1 ? counterfactual.player_1_action
                                                          : counterfactual.player_2_action;
    player_action = BlottoHistory::FogAction(new_action, [](...) { return true; });
    return terminal_value(counterfactual, player).value() - terminal_value(history, player).value();
}

std::ostream &operator<<(std::ostream &out, const BlottoHistory &history) {
    out << wise_enum::to_string(BlottoPlayer::PLAYER1) << "["
        << (history.player_1_action.has_value()
                ? wise_enum::to_string(history.player_1_action.value())
                : "")
        << "] ";
    out << wise_enum::to_string(BlottoPlayer::PLAYER2) << "["
        << (history.player_2_action.has_value()
                ? wise_enum::to_string(history.player_2_action.value())
                : "")
        << "] ";
    return out;
}
}  // namespace robot::domain
