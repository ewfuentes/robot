
#include "learning/cfr.hh"

#include "domain/rock_paper_scissors.hh"
#include "gtest/gtest.h"

namespace robot::learning {
TEST(CfrTest, rock_paper_scissors_test_fixed_opponent) {
    using RPS = domain::RockPaperScissors;
    const std::string INFO_SET_ID = "rps";
    // Setup
    const MinRegretTrainConfig<RPS> config = {
        .num_iterations = 100000,
        .fixed_strategies = {{RPS::Players::PLAYER2,
                              [](const RPS::History &) -> Strategy<RPS> {
                                  return {{RPS::Actions::ROCK, 0.4},
                                          {RPS::Actions::PAPER, 0.3},
                                          {RPS::Actions::SCISSORS, 0.3}};
                              }}},
        .info_set_id_from_hist = [&INFO_SET_ID](const RPS::History &) { return INFO_SET_ID; },
        .seed = 0,
    };

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<domain::RockPaperScissors>(config);

    // Verification
    const auto &strategy = min_regret_strategy.strategy_from_info_set_id.at(INFO_SET_ID);
    const double total_probability = strategy[RPS::Actions::ROCK] + strategy[RPS::Actions::PAPER] +
                                     strategy[RPS::Actions::SCISSORS];

    EXPECT_NEAR(total_probability, 1.0, 1e-6);
    EXPECT_LT(strategy[RPS::Actions::ROCK], 0.005);
    EXPECT_GT(strategy[RPS::Actions::PAPER], 0.99);
    EXPECT_LT(strategy[RPS::Actions::SCISSORS], 0.005);
}

TEST(CfrTest, rock_paper_scissors_test) {
    using RPS = domain::RockPaperScissors;
    const std::string INFO_SET_ID = "rps";
    // Setup
    const MinRegretTrainConfig<RPS> config = {
        .num_iterations = 100000,
        .fixed_strategies = {},
        .info_set_id_from_hist = [&INFO_SET_ID](const RPS::History &) { return INFO_SET_ID; },
        .seed = 0,
    };

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<domain::RockPaperScissors>(config);

    // Verification
    const auto &strategy = min_regret_strategy.strategy_from_info_set_id.at(INFO_SET_ID);
    const double total_probability = strategy[RPS::Actions::ROCK] + strategy[RPS::Actions::PAPER] +
                                     strategy[RPS::Actions::SCISSORS];

    EXPECT_NEAR(total_probability, 1.0, 1e-6);
    EXPECT_NEAR(strategy[RPS::Actions::ROCK], 0.333, 1e-2);
    EXPECT_NEAR(strategy[RPS::Actions::PAPER], 0.333, 1e-2);
    EXPECT_NEAR(strategy[RPS::Actions::SCISSORS], 0.333, 1e-2);
}
}  // namespace robot::learning
