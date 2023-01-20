

#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/time/robot_time.hh"
#include "domain/deck.hh"
#include "experimental/pokerbots/hand_evaluator.hh"

const std::filesystem::path BASE_DIR = "/tmp/cluster_hands";

namespace robot::experimental::pokerbots {

void evaluate_hole_cards() {
    const std::filesystem::path HOLE_CARDS_OUTPUT = BASE_DIR / "hole_cards_output.csv";
    std::cout << "Start!" << std::endl;
    // For every pair of hole cards, compute what the strength potential is
    const domain::StandardDeck deck;
    constexpr int hole_max_additional_cards = 7;
    constexpr int max_hands = 100000;
    std::mt19937 gen;

    {
        std::ofstream out(HOLE_CARDS_OUTPUT);
        out << "card1,card2,strength,neg_pot,pos_pot\n";
        const auto now = time::current_robot_time();
        for (auto iter_1 = deck.begin(); iter_1 != deck.end() - 1; iter_1++) {
            for (auto iter_2 = iter_1 + 1; iter_2 != deck.end(); iter_2++) {
                const auto result =
                    evaluate_strength_potential({*iter_1, *iter_2}, {}, hole_max_additional_cards,
                                                {}, max_hands, make_in_out(gen));
                out << to_string(*iter_1) << "," << to_string(*iter_2) << "," << result.strength
                    << "," << result.negative_potential << "," << result.positive_potential << "\n";
            }
        }
        const auto dt = time::current_robot_time() - now;
        std::cout << "time elapsed to evaluate all hole cards: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(dt) << std::endl;
    }

    // For every pair of hole cards and each possible board, compute what the strength potential is
    std::cout << "Done!" << std::endl;
}

void evaluate_flops() {
    const std::filesystem::path FLOP_CARDS_OUTPUT = BASE_DIR / "flop_cards";
    std::filesystem::create_directories(FLOP_CARDS_OUTPUT);
    std::cout << "Start!" << std::endl;
    // For every pair of hole cards, compute what the strength potential is
    const domain::StandardDeck deck;
    constexpr int hole_max_additional_cards = 5;
    constexpr int max_hands = 1000;

    std::vector<std::array<domain::StandardDeck::Card, 2>> hole_cards;
    for (auto iter_1 = deck.begin(); iter_1 != deck.end() - 1; iter_1++) {
        for (auto iter_2 = iter_1 + 1; iter_2 != deck.end(); iter_2++) {
            hole_cards.push_back({*iter_1, *iter_2});
        }
    }

    std::vector<std::vector<StrengthPotentialResult>> results;
    results.reserve(hole_cards.size());

    const auto now = time::current_robot_time();
    std::for_each(
        std::execution::par, hole_cards.begin(), hole_cards.end(), [&](const auto &cards) {
            const std::filesystem::path output_path =
                FLOP_CARDS_OUTPUT / (to_string(cards[0]) + to_string(cards[1]) + ".csv");
            std::mt19937 gen(std::hash<std::string>()(to_string(cards[0]) + to_string(cards[1])));
            const auto start = time::current_robot_time();
            std::ofstream out(output_path);
            out << "board1,board2,board3,strength,neg_pot,pos_pot\n";
            for (auto board_1 = deck.begin(); board_1 != deck.end() - 2; board_1++) {
                if (*board_1 == cards[0] || *board_1 == cards[1]) {
                    continue;
                }
                for (auto board_2 = board_1 + 1; board_2 != deck.end() - 1; board_2++) {
                    if (*board_2 == cards[0] || *board_2 == cards[1]) {
                        continue;
                    }
                    for (auto board_3 = board_2 + 1; board_3 != deck.end(); board_3++) {
                        if (*board_3 == cards[0] || *board_3 == cards[1]) {
                            continue;
                        }
                        const auto result = evaluate_strength_potential(
                            cards, {*board_1, *board_2, *board_3}, hole_max_additional_cards, {},
                            max_hands, make_in_out(gen));
                        out << to_string(*board_1) << "," << to_string(*board_2) << ","
                            << to_string(*board_3) << "," << result.strength << ","
                            << result.negative_potential << "," << result.positive_potential
                            << "\n";
                    }
                }
            }
            const auto dt = time::current_robot_time() - start;
            std::cout << to_string(cards[0]) << to_string(cards[1]) << " in "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(dt) << std::endl;
        });

    const auto dt = time::current_robot_time() - now;
    std::cout << "time elapsed to evaluate all hole cards: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(dt) << std::endl;

    // For every pair of hole cards and each possible board, compute what the strength potential is
    std::cout << "Done!" << std::endl;
}
}  // namespace robot::experimental::pokerbots

int main() {
    std::filesystem::create_directories(BASE_DIR);
    robot::experimental::pokerbots::evaluate_hole_cards();
    robot::experimental::pokerbots::evaluate_flops();
}
