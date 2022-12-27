
#include "domain/rock_paper_scissors.hh"

#include <iostream>

namespace robot::domain {

RPSAction get_input(const RPSPlayer player) {
  if (player == RPSPlayer::PLAYER1) {
    return RPSAction::ROCK;
  }
  return RPSAction::SCISSORS;
}

void play_game() {
    RPSHistory history;

    while (true) {
        const auto maybe_player = up_next(history);
        if (!maybe_player.has_value()) {
            break;
        }
        const RPSAction action = get_input(maybe_player.value());
        history = play(history, action);
    }

    const int player_1_terminal_value = terminal_value(history, RPSPlayer::PLAYER1).value();

    if (player_1_terminal_value == 1) {
        std::cout << "Player 1 wins!" << std::endl;
    } else if (player_1_terminal_value == 0) {
        std::cout << "Tie!" << std::endl;
    } else {
        std::cout << "Player 2 wins!" << std::endl;
    }
}

}  // namespace robot::domain

int main(const int argc, const char **argv) {
    (void)argc;
    (void)argv;
    robot::domain::play_game();
}
