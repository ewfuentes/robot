
#include "learning/min_regret_strategy_to_proto.hh"

#include <iostream>

#include "domain/rock_paper_scissors.hh"
#include "gtest/gtest.h"
#include "learning/cfr.hh"

namespace robot::learning::proto {
TEST(MinRegretStrategyToProtoTest, happy_case) {
    // Setup
    using RPS = domain::RockPaperScissors;
    const std::unordered_map<std::string, learning::InfoSetCounts<RPS>> counts_from_id = {
        {"RPS",
         {.regret_sum =
              {
                  {RPS::Actions::ROCK, 2.0},
                  {RPS::Actions::PAPER, 3.0},
                  {RPS::Actions::SCISSORS, 4.0},
              },
          .strategy_sum =
              {
                  {RPS::Actions::ROCK, 10.0},
                  {RPS::Actions::PAPER, 11.0},
                  {RPS::Actions::SCISSORS, 12.0},
              },
          .iter_count = 123}},
        {"ABC",
         {.regret_sum =
              {
                  {RPS::Actions::ROCK, 20.0},
                  {RPS::Actions::PAPER, 30.0},
                  {RPS::Actions::SCISSORS, 40.0},
              },
          .strategy_sum =
              {
                  {RPS::Actions::ROCK, 210.0},
                  {RPS::Actions::PAPER, 211.0},
                  {RPS::Actions::SCISSORS, 212.0},
              },
          .iter_count = 1230}},
    };

    // Action
    MinRegretStrategy proto;
    pack_into<RPS>(counts_from_id, &proto);
    const auto unpacked_counts = unpack_from<RPS>(proto);

    // Verification
    EXPECT_EQ(unpacked_counts.size(), counts_from_id.size());
    for (const auto &[id, counts] : counts_from_id) {
        const auto iter = unpacked_counts.find(id);
        EXPECT_NE(iter, unpacked_counts.end());
        EXPECT_EQ(iter->second.iter_count, counts.iter_count);

        for (const auto &[action, name] : Range<RPS::Actions>::value) {
            EXPECT_EQ(iter->second.regret_sum[action], counts.regret_sum[action]);
            EXPECT_EQ(iter->second.strategy_sum[action], counts.strategy_sum[action]);
        }
    }
}
}  // namespace robot::learning::proto
