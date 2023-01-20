
#pragma once

#include <random>

#include "common/argument_wrapper.hh"
#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/bin_centers.pb.h"

namespace robot::experimental::pokerbots {
domain::RobPoker::InfoSetId infoset_id_from_history(const domain::RobPokerHistory &history,
                                                    const proto::PerTurnBinCenters &bin_centers,
                                                    InOut<std::mt19937> gen);

domain::RobPoker::InfoSetId infoset_id_from_information(
    const std::array<domain::StandardDeck::Card, 2> &private_cards,
    const std::vector<domain::StandardDeck::Card> &common_cards,
    const std::vector<domain::RobPokerAction> &actions, const domain::BettingState &betting_state,
    const proto::PerTurnBinCenters &bin_centers, InOut<std::mt19937> gen);
}  // namespace robot::experimental::pokerbots
