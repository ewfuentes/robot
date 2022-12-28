
#include <iostream>
#include <limits>

#include "domain/rock_paper_scissors.hh"

namespace robot::domain {

std::optional<RPSAction> get_input(const RPSPlayer player) {
    std::cout << "Enter action for player " << (player == RPSPlayer::PLAYER1 ? "1" : "2")
              << std::endl;
    std::cout << "{1: Rock, 2: Paper, 3: Scissors}" << std::endl;
    int action;
    std::cin >> action;
    if (std::cin.fail() || action < 1 || action > 3) {
        std::cout << "Invalid Input!" << std::endl;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return std::nullopt;
    }

    if (action == 1) {
        return RPSAction::ROCK;
    } else if (action == 2) {
        return RPSAction::PAPER;
    } else {
        return RPSAction::SCISSORS;
    }
}

void play_game() {
    RPSHistory history;

    while (true) {
        const auto maybe_player = up_next(history);
        if (!maybe_player.has_value()) {
            break;
        }
        const std::optional<RPSAction> maybe_action = get_input(maybe_player.value());
        if (maybe_action.has_value()) {
            history = play(history, maybe_action.value());
        }
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
