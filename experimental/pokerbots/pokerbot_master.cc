
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <random>
#include <string_view>
#include <thread>
#include <variant>

#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "domain/rob_poker.hh"
#include "experimental/pokerbots/bin_centers_to_proto.hh"
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

std::vector<domain::RobPokerAction> action_generator(const domain::RobPokerHistory &history) {
    const domain::BettingState betting_state = domain::compute_betting_state(history);
    std::vector<domain::RobPokerAction> out;
    out.reserve(RobPokerActionIndex::size);
    for (const auto &action : domain::possible_actions(history)) {
        if (std::holds_alternative<domain::RaiseAction>(action)) {
            const int max_raise = std::get<domain::RaiseAction>(action).amount;
            const int pot_size = betting_state.put_in_pot[domain::RobPokerPlayer::PLAYER1] +
                                 betting_state.put_in_pot[domain::RobPokerPlayer::PLAYER2];
            const int max_actions = betting_state.to_bet->round == 0 ? 6 : 4;
            if (pot_size < max_raise && betting_state.to_bet->position < max_actions) {
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

int train(const std::filesystem::path &output_directory, const uint64_t num_iterations,
          const std::filesystem::path &bin_centers_path,
          const std::optional<std::filesystem::path> &maybe_load_checkpoint) {
    // create the ouput directory
    std::filesystem::create_directories(output_directory);

    // Load the bin centers file
    if (!std::filesystem::exists(bin_centers_path)) {
        std::cout << "Unable to find bin centers file at: " << bin_centers_path << std::endl;
        return 1;
    }

    const PerTurnBinCenters bin_centers = [&bin_centers_path]() {
        proto::PerTurnBinCenters proto;
        std::ifstream file_in(bin_centers_path, std::ios_base::binary);
        proto.ParseFromIstream(&file_in);
        return unpack_from(proto);
    }();

    const learning::MinRegretTrainConfig<RobPoker> config = {
        .num_iterations = num_iterations,
        .infoset_id_from_hist =
            [&bin_centers](const RobPoker::History &history) {
                std::mt19937 gen(0);
                return infoset_id_from_history(history, bin_centers, make_in_out(gen));
            },
        .action_generator = action_generator,
        .seed = 0,
        .num_threads = 20,
        .sample_strategy = learning::SampleStrategy::EXTERNAL_SAMPLING,
        .iteration_callback =
            [prev_t = std::optional<time::RobotTimestamp>{}, &output_directory,
             &maybe_load_checkpoint](const int iter, auto counts_from_infoset_id) mutable {
                constexpr int ITERS_BETWEEN_DISCOUNTS = 1000000;
                constexpr int ITERS_BETWEEN_SAVES = 1000000;
                if (iter == 0 && maybe_load_checkpoint.has_value()) {
                    if (std::filesystem::exists(maybe_load_checkpoint.value())) {
                        // load the existing checkpoint
                        learning::proto::MinRegretStrategy proto;
                        std::ifstream in(*maybe_load_checkpoint, std::ios_base::binary);
                        proto.ParseFromIstream(&in);
                        *counts_from_infoset_id = unpack_from<RobPoker>(proto);
                    } else {
                        std::cout << maybe_load_checkpoint.value()
                                  << " does not exist! Skipping checkpoint load" << std::endl;
                    }
                }

                const auto now = time::current_robot_time();
                if (prev_t.has_value()) {
                    const auto dt = now - prev_t.value();
                    const auto dt_s = std::chrono::duration<double>(dt).count();
                    std::cout << iter << " dt: " << dt_s
                              << " sec samples/sec:" << ITERS_BETWEEN_SAVES / dt_s << std::endl;
                }
                prev_t = now;

                std::stringstream idx;
                idx << std::setfill('0') << std::setw(9) << iter;
                const auto path = output_directory / ("pokerbot_checkpoint_" + idx.str() + ".pb");

                learning::proto::MinRegretStrategy proto;
                pack_into(*counts_from_infoset_id, &proto);
                std::ofstream out(path, std::ios_base::binary | std::ios_base::trunc);
                proto.SerializeToOstream(&out);

                const int num_periods = iter / ITERS_BETWEEN_DISCOUNTS;
                const double factor = num_periods / (num_periods + 1.0);
                for (auto &[_, counts] : *counts_from_infoset_id) {
                    for (auto &[action, _] : Range<domain::RobPokerAction>::value) {
                        counts.regret_sum[action] *= factor;
                        counts.strategy_sum[action] *= factor;
                    }
                }
                return ITERS_BETWEEN_SAVES;
            },
    };
    const auto t_start = time::current_robot_time();
    train_min_regret_strategy<RobPoker>(config);
    const auto dt = time::current_robot_time() - t_start;
    const auto dt_s = std::chrono::duration<double>(dt).count();
    std::cout << "total time: " << dt_s << std::endl;
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
        std::cout << std::chrono::duration<double>(item).count() << ", ";
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
    if (betting_state.to_bet->position == 0) {
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
      ("bins", "Path to hand bins file", cxxopts::value<std::string>())
      ("load_checkpoint", "Path to checkpoint file to continue", cxxopts::value<std::string>())
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
    if (args.count("bins") == 0) {
        std::cout << "bins is a required option" << std::endl;
        std::exit(1);
    }

    return robot::experimental::pokerbots::train(
        args["output_directory"].as<std::string>(), args["num_iterations"].as<uint64_t>(),
        args["bins"].as<std::string>(),
        args.count("load_checkpoint")
            ? std::make_optional(args["load_checkpoint"].as<std::string>())
            : std::nullopt);
}
