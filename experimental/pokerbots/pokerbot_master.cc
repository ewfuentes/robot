
#include <chrono>
#include <string_view>
#include <variant>

#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/generate_infoset_id.hh"
#include "learning/cfr.hh"

namespace robot {
namespace {
constexpr std::tuple<domain::RobPokerAction, std::string_view> make_index_item(const auto action) {
    using T = std::decay_t<decltype(action)>;
    return {T{}, T::name};
}
}  // namespace

struct RobPokerActionIndex {
    static constexpr std::array<std::tuple<domain::RobPokerAction, std::string_view>, 5> index{{
        make_index_item(domain::FoldAction{}),
        make_index_item(domain::CheckAction{}),
        make_index_item(domain::CallAction{}),
        make_index_item(domain::RaisePotAction{}),
        make_index_item(domain::AllInAction{}),
    }};
    static constexpr int size = index.size();
};

template <>
struct IndexSize<domain::RobPokerAction> {
    static constexpr int value = RobPokerActionIndex::size;
};

template <>
struct Range<domain::RobPokerAction> {
    static constexpr auto value = RobPokerActionIndex::index;
};

namespace experimental::pokerbots {
using RobPoker = domain::RobPoker;

std::vector<domain::RobPokerAction> action_generator(const domain::RobPokerHistory &history) {
    const domain::BettingState betting_state = domain::compute_betting_state(history);
    std::vector<domain::RobPokerAction> out;
    out.reserve(RobPokerActionIndex::size);
    for (const auto &action : domain::possible_actions(history)) {
        if (std::holds_alternative<domain::RaiseAction>(action)) {
            const int max_raise = std::get<domain::RaiseAction>(action).amount;
            const int pot_size = betting_state.put_in_pot[domain::RobPokerPlayer::PLAYER1] +
                                 betting_state.put_in_pot[domain::RobPokerPlayer::PLAYER2];
            const int max_actions = betting_state.round == 0 ? 6 : 4;
            if (pot_size < max_raise && betting_state.position < max_actions) {
                out.push_back(domain::RaisePotAction{});
            }
            out.push_back(domain::AllInAction{});
        } else {
            out.push_back(action);
        }
    }
    return out;
}

int train() {
    const learning::MinRegretTrainConfig<RobPoker> config = {
        .num_iterations = 1000,
        .infoset_id_from_hist =
            [](const RobPoker::History &history) { return infoset_id_from_history(history); },
        .action_generator = action_generator,
        .seed = 0,
        .sample_strategy = learning::SampleStrategy::EXTERNAL_SAMPLING,
        .iteration_callback =
            [prev_t = std::optional<time::RobotTimestamp>{}](
                const int iter, const auto &counts_from_infoset_id) mutable {
                (void)counts_from_infoset_id;
                constexpr int ITERS_BETWEEN_PRINTS = 20;
                if (iter % ITERS_BETWEEN_PRINTS == 0) {
                    const auto now = time::current_robot_time();
                    if (prev_t.has_value()) {
                      const auto dt = now - prev_t.value();
                      const auto dt_s = std::chrono::duration<double>(dt).count();
                        std::cout << iter << " dt: " << dt_s
                                  << " sec samples/sec:" << ITERS_BETWEEN_PRINTS / dt_s
                                  << std::endl;
                    }
                    prev_t = now;
                }
                return true;
            },
    };
    train_min_regret_strategy<RobPoker>(config);
    return 0;
}

uint64_t count_infosets(const domain::RobPokerHistory &history = {}) {
    const auto maybe_next_player = up_next(history);
    if (!maybe_next_player.has_value()) {
        return 1;
    } else if (maybe_next_player.value() == domain::RobPokerPlayer::CHANCE) {
        std::mt19937 gen;
        return count_infosets(play(history, make_in_out(gen)).history);
    }
    const auto betting_state = domain::compute_betting_state(history);
    if (betting_state.position == 0) {
        return 1;
    }

    uint64_t num_actions = 0;
    for (const auto &action : action_generator(history)) {
        num_actions += count_infosets(play(history, action));
    }
    return num_actions;
}

}  // namespace experimental::pokerbots
}  // namespace robot

int main(int argc, char **argv) {
    cxxopts::Options options("cfr_train", "I wanna be the very best, like no one ever was.");
    auto args = options.parse(argc, argv);

    return robot::experimental::pokerbots::train();
}
