
#pragma once

#include "domain/rob_poker.hh"
#include "domain/deck.hh"

namespace robot::experimental::pokerbots {
domain::RobPoker::InfoSetId infoset_id_from_history(const domain::RobPokerHistory &history);
domain::RobPoker::InfoSetId infoset_id_from_information(
    const std::array<domain::StandardDeck::Card, 2> &private_cards,
    const std::vector<domain::StandardDeck::Card> &common_cards,
    const std::vector<domain::RobPokerAction> &actions, const domain::BettingState &betting_state);
}  // namespace robot::experimental::pokerbots
