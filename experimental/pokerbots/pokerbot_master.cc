
#include <string_view>
#include <variant>

#include "cxxopts.hpp"
#include "domain/rob_poker.hh"
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
        .num_iterations = 100000,
        .infoset_id_from_hist = [](const RobPoker::History &) { return RobPoker::InfoSetId{}; },
        .action_generator = action_generator,
        .seed = 0,
        .sample_strategy = learning::SampleStrategy::EXTERNAL_SAMPLING,
    };
    train_min_regret_strategy<RobPoker>(config);
    return 0;
}

}  // namespace experimental::pokerbots
}  // namespace robot

int main(int argc, char **argv) {
    cxxopts::Options options("cfr_train", "I wanna be the very best, like no one ever was.");
    auto args = options.parse(argc, argv);

    return robot::experimental::pokerbots::train();
}
