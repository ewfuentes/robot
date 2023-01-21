
#include "experimental/pokerbots/generate_infoset_id.hh"

#include "domain/deck.hh"
#include "domain/rob_poker.hh"
#include "gtest/gtest.h"

namespace robot::experimental::pokerbots {
namespace {
using Card = domain::StandardDeck::Card;
using Suits = domain::StandardDeck::Card::Suits;
using Ranks = domain::StandardDeck::Card::Ranks;
}  // namespace
TEST(GenerateInfosetIdTest, basic_test) {
    // Setup
    const proto::PerTurnBinCenters bin_centers{};
    const domain::BettingState betting_state = {
        .put_in_pot = {0},
        .is_game_over = false,
        .showdown_required = false,
        .last_played = domain::RobPokerPlayer::PLAYER2,
        .to_bet = domain::BettingState::ToBet{
            .is_final_betting_round = false, .round = 0, .position = 1}};
    std::mt19937 gen(0);

    // Action
    const auto infoset_id = infoset_id_from_information(
        {Card(Ranks::_A, Suits::HEARTS), Card(Ranks::_A, Suits::DIAMONDS)}, {}, {}, betting_state,
        bin_centers, make_in_out(gen));
    // Verification
    std::cout << std::hex << infoset_id << std::endl;
}

}  // namespace robot::experimental::pokerbots
