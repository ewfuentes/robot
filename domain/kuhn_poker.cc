
#include "domain/kuhn_poker.hh"

#include <algorithm>
#include <random>
#include <sstream>

namespace robot::domain {
namespace {
bool is_showdown(const std::vector<KuhnAction> &actions) {
    if (actions.size() == 2 && actions.at(0) == actions.at(1)) {
        // Both players passed or bet
        return true;
    } else if (actions.size() == 3 && actions.back() == KuhnAction::BET) {
        // Player 1 bet after passing
        return true;
    }
    // A player folded, so no showdown is required
    return false;
}

bool operator<(const KuhnHistory::Card a, const KuhnHistory::Card b) {
    return static_cast<int>(a) < static_cast<int>(b);
}

}  // namespace
KuhnHistory play(const KuhnHistory &history, const KuhnAction &action) {
    const auto maybe_next_player = up_next(history);
    if (!maybe_next_player.has_value() || maybe_next_player.value() == KuhnPlayer::CHANCE) {
        // This function can only be called if Player 1 or Player 2 must act
        return history;
    }
    KuhnHistory out = history;
    out.actions.push_back(action);
    return out;
}

ChanceResult play(const KuhnHistory &history, InOut<std::mt19937> gen) {
    using Card = KuhnHistory::Card;
    std::vector<Card> deck{Card::A, Card::K, Card::Q};
    KuhnHistory out = history;

    // Remove any cards from the deck that may have already been dealt
    for (const auto player : {KuhnPlayer::PLAYER1, KuhnPlayer::PLAYER2}) {
        if (out.hands[player].has_value()) {
            std::erase(deck, out.hands[player].value());
        }
    }

    std::shuffle(deck.begin(), deck.end(), *gen);

    double probability = 1.0;
    for (const auto player : {KuhnPlayer::PLAYER1, KuhnPlayer::PLAYER2}) {
        if (!out.hands[player].has_value()) {
            out.hands[player] = KuhnHistory::FogCard(
                deck.back(),
                [curr_player = player](const auto player) { return player == curr_player; });
            probability *= 1.0 / deck.size();
            deck.pop_back();
        }
    }

    return {.history = std::move(out), .probability = probability};
}

std::optional<KuhnPlayer> up_next(const KuhnHistory &history) {
    if (!history.hands[KuhnPlayer::PLAYER1].has_value() ||
        !history.hands[KuhnPlayer::PLAYER2].has_value()) {
        // Some cards haven't been dealt yet, deal before continuing
        return KuhnPlayer::CHANCE;
    } else if (history.actions.empty()) {
        // No Actions have been played, Player 1 goes first
        return KuhnPlayer::PLAYER1;
    } else if (history.actions.size() == 1) {
        // Only a single action has been played, Player 2 goes next
        return KuhnPlayer::PLAYER2;
    } else if (history.actions.size() == 2 && history.actions.at(0) == KuhnAction::PASS &&
               history.actions.at(1) == KuhnAction::BET) {
        // If player 1 has passed and player 2 has bet, Player 1 must respond
        return KuhnPlayer::PLAYER1;
    }
    // The game is over
    return std::nullopt;
}

std::vector<KuhnAction> possible_actions(const KuhnHistory &history) {
    const auto maybe_next_player = up_next(history);
    if (!maybe_next_player.has_value() || maybe_next_player.value() == KuhnPlayer::CHANCE) {
        return {};
    }
    return {KuhnAction::PASS, KuhnAction::BET};
}

std::optional<int> terminal_value(const KuhnHistory &history, const KuhnPlayer player) {
    const auto maybe_next_player = up_next(history);
    if (maybe_next_player.has_value()) {
        return std::nullopt;
    }

    if (is_showdown(history.actions)) {
        // Given that a showdown is required and player 2 bet, then the value is 2
        const int value = history.actions.at(1) == KuhnAction::BET ? 2 : 1;
        const KuhnPlayer opponent =
            player == KuhnPlayer::PLAYER1 ? KuhnPlayer::PLAYER2 : KuhnPlayer::PLAYER1;
        const int sign = history.hands[player].value() < history.hands[opponent].value() ? -1 : 1;
        return sign * value;
    } else {
        // No showdown required, if player 1 bet, then player 1 wins
        const int player_1_value = history.actions.at(0) == KuhnAction::BET ? 1 : -1;
        const int sign = player == KuhnPlayer::PLAYER1 ? 1 : -1;
        return sign * player_1_value;
    }
}

KuhnPoker::InfoSetId infoset_id_from_history(const KuhnHistory &history) {
    // Pack private card
    const auto current_player = up_next(history).value();
    const auto private_card = history.hands[current_player].value();
    return infoset_id_from_information(private_card, history.actions);
}

KuhnPoker::InfoSetId infoset_id_from_information(const KuhnHistory::Card private_card,
                                                 const std::vector<KuhnAction> &actions) {
    int out = 0;
    // Pack actions
    for (const auto action : actions) {
        out = (out << 2) | (static_cast<int>(action) + 1);
    }

    // Pack private card
    out = (out << 2) | (static_cast<int>(private_card) + 1);

    return out;
}

std::string to_string(const KuhnHistory &history) {
    const auto current_player = up_next(history).value();
    std::stringstream id;
    // Serialize the private information
    id << wise_enum::to_string(history.hands[current_player].value());
    if (!history.actions.empty()) {
        id << "_[";
        // Serialize the public information
        for (auto action : history.actions) {
            id << wise_enum::to_string(action) << ",";
        }
        id.seekp(-1, std::ios_base::end);
        id << "]";
    }
    return id.str();
}

std::ostream &operator<<(std::ostream &out, const KuhnHistory &history) {
    out << to_string(history);
    return out;
}

}  // namespace robot::domain
