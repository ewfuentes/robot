
#include "domain/tiger.hh"

#include "gtest/gtest.h"

namespace robot::domain {
TEST(TigerTest, picking_listen_returns_observation) {
    // SETUP
    const TigerConfig config = {
        .consistent_observation_probability = 0.85,
        .listening_reward = -1.0,
        .treasure_reward = 10.0,
        .tiger_reward = -100.0,
    };
    Tiger tiger(config);

    // ACTION
    const Tiger::Result result = tiger.step(Tiger::Action::LISTEN);

    // VERIFICATION
    EXPECT_EQ(result.reward, config.listening_reward);
    EXPECT_NE(result.observation, Tiger::Observation::TREASURE);
    EXPECT_NE(result.observation, Tiger::Observation::TIGER);
    EXPECT_NE(result.observation, Tiger::Observation::INVALID);
}

TEST(TigerTest, opening_to_tiger_yields_tiger_observation) {
    // SETUP
    const TigerConfig config = {
        .consistent_observation_probability = 0.85,
        .listening_reward = -1.0,
        .treasure_reward = 10.0,
        .tiger_reward = -100.0,
    };
    Tiger tiger(config);

    // ACTION
    const Tiger::Action action =
        tiger.is_tiger_left() ? Tiger::Action::OPEN_LEFT : Tiger::Action::OPEN_RIGHT;
    const Tiger::Result result = tiger.step(action);

    // VERIFICATION
    EXPECT_EQ(result.reward, config.tiger_reward);
    EXPECT_EQ(result.observation, Tiger::Observation::TIGER);
    EXPECT_TRUE(result.is_done);
}

TEST(TigerTest, opening_to_treasure_yields_treasure_observation) {
    // SETUP
    const TigerConfig config = {
        .consistent_observation_probability = 0.85,
        .listening_reward = -1.0,
        .treasure_reward = 10.0,
        .tiger_reward = -100.0,
    };
    Tiger tiger(config);

    // ACTION
    const Tiger::Action action =
        tiger.is_tiger_left() ? Tiger::Action::OPEN_RIGHT : Tiger::Action::OPEN_LEFT;
    const Tiger::Result result = tiger.step(action);

    // VERIFICATION
    EXPECT_EQ(result.reward, config.treasure_reward);
    EXPECT_EQ(result.observation, Tiger::Observation::TREASURE);
    EXPECT_TRUE(result.is_done);
}
}  // namespace robot::domain
