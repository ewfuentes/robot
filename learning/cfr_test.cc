
#include "learning/cfr.hh"

#include "domain/blotto.hh"
#include "domain/kuhn_poker.hh"
#include "domain/rock_paper_scissors.hh"
#include "gtest/gtest.h"

namespace robot::learning {

class CfrTest : public testing::Test, public testing::WithParamInterface<SampleStrategy> {};

TEST_P(CfrTest, rock_paper_scissors_test) {
    using RPS = domain::RockPaperScissors;
    const std::string INFOSET_ID = "rps";
    // Setup
    const uint64_t external_iter_scale_factor =
        GetParam() == SampleStrategy::EXTERNAL_SAMPLING ? 10 : 1;
    const uint64_t num_iterations = 10000 * external_iter_scale_factor;
    const MinRegretTrainConfig<RPS> config = {
        .num_iterations = num_iterations,
        .infoset_id_from_hist = [&INFOSET_ID](const RPS::History &) { return INFOSET_ID; },
        .action_generator =
            [](const RPS::History &history) { return domain::possible_actions(history); },
        .seed = 0,
        .sample_strategy = GetParam(),
        .iteration_callback = [&](const auto &, const auto &) { return num_iterations; },
    };

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<RPS>(config);

    // Verification
    const auto &strategy = min_regret_strategy.strategy_from_infoset_id.at(INFOSET_ID);
    const double total_probability = strategy[RPS::Actions::ROCK] + strategy[RPS::Actions::PAPER] +
                                     strategy[RPS::Actions::SCISSORS];

    EXPECT_NEAR(total_probability, 1.0, 1e-6);
    EXPECT_NEAR(strategy[RPS::Actions::ROCK], 0.333, 1e-2);
    EXPECT_NEAR(strategy[RPS::Actions::PAPER], 0.333, 1e-2);
    EXPECT_NEAR(strategy[RPS::Actions::SCISSORS], 0.333, 1e-2);
}

TEST_P(CfrTest, blotto_test) {
    using Blotto = domain::Blotto;
    const std::string INFOSET_ID = "blotto";
    // Setup
    const uint64_t external_iter_scale_factor =
        GetParam() == SampleStrategy::EXTERNAL_SAMPLING ? 10 : 1;
    const uint64_t num_iterations = 10000 * external_iter_scale_factor;
    const MinRegretTrainConfig<Blotto> config = {
        .num_iterations = num_iterations,
        .infoset_id_from_hist = [&INFOSET_ID](const Blotto::History &) { return INFOSET_ID; },
        .action_generator =
            [](const Blotto::History &history) { return domain::possible_actions(history); },
        .seed = 0,
        .sample_strategy = GetParam(),
        .iteration_callback = [&](const auto &, const auto &) { return num_iterations; },
    };

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<Blotto>(config);

    // Verification
    const auto &strategy = min_regret_strategy.strategy_from_infoset_id.at(INFOSET_ID);

    const double total_probability = std::accumulate(
        strategy.begin(), strategy.end(), 0.0,
        [](const double sum, const auto &action_and_prob) { return sum + action_and_prob.second; });
    EXPECT_NEAR(total_probability, 1.0, 1e-6);
    // These strategies are easily dominated, will only win 1 in the best case, so they should have
    // very low probability
    EXPECT_NEAR(strategy[Blotto::Actions::ASSIGN_500], 0.0, 1e-6);
    EXPECT_NEAR(strategy[Blotto::Actions::ASSIGN_050], 0.0, 1e-6);
    EXPECT_NEAR(strategy[Blotto::Actions::ASSIGN_005], 0.0, 1e-6);
}

TEST_P(CfrTest, kuhn_poker_test) {
    using KuhnPoker = domain::KuhnPoker;
    using Card = domain::KuhnPoker::History::Card;
    using Action = domain::KuhnPoker::Actions;

    // Setup
    constexpr double TOL = 1e-2;
    const uint64_t external_iter_scale_factor =
        GetParam() == SampleStrategy::EXTERNAL_SAMPLING ? 10 : 1;
    const uint64_t num_iterations = 200000 * external_iter_scale_factor;
    const MinRegretTrainConfig<KuhnPoker> config = {
        .num_iterations = num_iterations,
        .infoset_id_from_hist = &domain::infoset_id_from_history,
        .action_generator =
            [](const domain::KuhnHistory &history) { return domain::possible_actions(history); },
        .seed = 0,
        .sample_strategy = GetParam(),
        .iteration_callback = [&](const auto &, const auto &) { return num_iterations; },
    };

    // Action
    const auto min_regret_strategy = train_min_regret_strategy<KuhnPoker>(config);

    // Verification
    // Spot check a few strategies
    {
        // You should always bet with an Ace after a bet
        const auto maybe_strategy =
            min_regret_strategy(domain::infoset_id_from_information(Card::A, {Action::BET}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_NEAR(maybe_strategy.value()[Action::BET], 1.0, TOL);
    }
    {
        // You should always pass with an queen after a bet
        const auto maybe_strategy =
            min_regret_strategy(domain::infoset_id_from_information(Card::Q, {Action::BET}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_NEAR(maybe_strategy.value()[Action::PASS], 1.0, TOL);
    }
    {
        // You should always bet with an Ace after a pass
        const auto maybe_strategy =
            min_regret_strategy(domain::infoset_id_from_information(Card::A, {Action::PASS}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_NEAR(maybe_strategy.value()[Action::BET], 1.0, TOL);
    }
    {
        // You should always bet with an Ace after a pass/bet
        const auto maybe_strategy = min_regret_strategy(
            domain::infoset_id_from_information(Card::A, {Action::PASS, Action::BET}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_NEAR(maybe_strategy.value()[Action::BET], 1.0, TOL);
    }
    {
        // You should sometimes bluff with a queen
        const auto maybe_strategy =
            min_regret_strategy(domain::infoset_id_from_information(Card::Q, {}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_GT(maybe_strategy.value()[Action::BET], TOL);
    }
    {
        // You should sometimes bluff a weak hand with an Ace
        const auto maybe_strategy =
            min_regret_strategy(domain::infoset_id_from_information(Card::A, {}));
        EXPECT_TRUE(maybe_strategy.has_value());
        EXPECT_GT(maybe_strategy.value()[Action::PASS], TOL);
    }
}

INSTANTIATE_TEST_SUITE_P(SampleStrategies, CfrTest,
                         testing::Values(SampleStrategy::CHANCE_SAMPLING,
                                         SampleStrategy::EXTERNAL_SAMPLING),
                         [](const auto &info) {
                             return std::string(wise_enum::to_string(info.param));
                         });
}  // namespace robot::learning
