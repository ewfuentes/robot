
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string_view>
#include <variant>

#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/generate_infoset_id.hh"
#include "learning/cfr.hh"
#include "learning/min_regret_strategy_to_proto.hh"

namespace robot {
namespace domain {
extern std::array<uint64_t, 33> eval_counts;
extern std::array<uint64_t, 33> board_sizes;
extern std::array<time::RobotTimestamp::duration, 33> eval_time;
extern std::array<time::RobotTimestamp::duration, 33> max_eval_time;
}  // namespace domain
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
extern uint64_t low_counts;
extern uint64_t high_counts;

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

extern std::array<uint64_t, 33> eval_strength_counts;
extern std::array<uint64_t, 33> eval_strength_time;

int train(const std::filesystem::path &output_directory, const uint64_t num_iterations) {
    const learning::MinRegretTrainConfig<RobPoker> config = {
        .num_iterations = num_iterations,
        .infoset_id_from_hist =
            [](const RobPoker::History &history) { return infoset_id_from_history(history); },
        .action_generator = action_generator,
        .seed = 0,
        .sample_strategy = learning::SampleStrategy::EXTERNAL_SAMPLING,
        .iteration_callback =
            [prev_t = std::optional<time::RobotTimestamp>{}, &output_directory](
                const int iter, const auto &counts_from_infoset_id) mutable {
                (void)counts_from_infoset_id;
                constexpr int ITERS_BETWEEN_PRINTS = 10000;
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

                constexpr int ITERS_BETWEEN_SAVES = 10000;
                if (iter % ITERS_BETWEEN_SAVES == 0) {
                    const auto path =
                        output_directory / ("pokerbot_checkpoint_" + std::to_string(iter) + ".pb");

                    learning::proto::MinRegretStrategy proto;
                    pack_into(counts_from_infoset_id, &proto);
                    std::ofstream out(path, std::ios_base::binary | std::ios_base::trunc);
                    proto.SerializeToOstream(&out);
                }
                return true;
            },
    };
    const auto t_start = time::current_robot_time();
    train_min_regret_strategy<RobPoker>(config);
    const auto dt = time::current_robot_time() - t_start;
    const auto dt_s = std::chrono::duration<double>(dt).count();
    std::cout << "total time: " << dt_s << std::endl;
    std::cout << low_counts << " " << high_counts << std::endl;
    std::cout << "board_sizes [";
    for (const auto &item : domain::board_sizes) {
        std::cout << item << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "evaluate_hand [";
    for (const auto &item : domain::eval_counts) {
        std::cout << item << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "evaluate_hand time [";
    double eval_hand_time_s = 0;
    for (const auto &item : domain::eval_time) {
        eval_hand_time_s += std::chrono::duration<double>(item).count();
        std::cout << std::chrono::duration<double>(item).count() << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "eval hand time total: " << eval_hand_time_s << std::endl;
    std::cout << "evaluate_hand time max [";
    for (const auto &item : domain::max_eval_time) {
        std::cout << std::chrono::duration<double>(item) << ", ";
    }
    std::cout << "]" << std::endl;
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
    // clang-format off
    options.add_options()
      ("output_directory", "Directory where checkpoints should be written to", cxxopts::value<std::string>())
      ("num_iterations", "Number of iterations", cxxopts::value<uint64_t>())
      ("help", "Print usage");
    // clang-format on
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    if (args.count("output_directory") == 0) {
        std::cout << "output_directory is a required option" << std::endl;
        std::exit(1);
    }
    if (args.count("num_iterations") == 0) {
        std::cout << "num_iterations is a required option" << std::endl;
        std::exit(1);
    }

    return robot::experimental::pokerbots::train(args["output_directory"].as<std::string>(),
                                                 args["num_iterations"].as<uint64_t>());
}
