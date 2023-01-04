
#include "learning/cfr.hh"

#include "domain/rock_paper_scissors.hh"
#include "gtest/gtest.h"

namespace robot::learning {
TEST(CfrTest, rock_paper_scissors_test_fixed_opponent) {
    using RPS = domain::RockPaperScissors;
    // Setup
    const MinRegretTrainConfig<RPS> config = {
        .num_iterations = 100,
        .fixed_strategies = {{RPS::Players::PLAYER2, [](const RPS::History &) -> Strategy<RPS> {
                                  return {{RPS::Actions::ROCK, 0.4},
                                          {RPS::Actions::PAPER, 0.3},
                                          {RPS::Actions::SCISSORS, 0.3}};
                              }}}};

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<domain::RockPaperScissors>(config);

    // Verification
    EXPECT_FALSE(true);
}
}  // namespace robot::learning
