
#include <variant>

#include "cxxopts.hpp"
#include "domain/rob_poker.hh"
#include "learning/cfr.hh"

namespace robot {

struct RobPokerActionIndex {
    static constexpr std::array<std::tuple<domain::RobPokerAction, std::string_view>, 8> index{
        {
            {domain::FoldAction{}, domain::FoldAction::name},
            {domain::CheckAction{}, domain::CheckAction::name},
            {domain::CallAction{}, domain::CallAction::name},
            {domain::RaiseAction{20}, "Raise20"},
            {domain::RaiseAction{40}, "Raise40"},
            {domain::RaiseAction{80}, "Raise80"},
            {domain::RaiseAction{160}, "Raise160"},
        }
        // action_element(domain::FoldAction{}),    action_element(domain::CheckAction{}),
        //        action_element(domain::CallAction{}),    action_element(domain::RaiseAction{5}),
        //        action_element(domain::RaiseAction{10}), action_element(domain::RaiseAction{20}),
        //        action_element(domain::RaiseAction{40}), action_element(domain::RaiseAction{80}),
    };
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
            if (betting_state.position > 5) {
                continue;
            }
            const int max_raise = std::get<domain::RaiseAction>(action).amount;
            for (const auto &[possible_action, _] : RobPokerActionIndex::index) {
                if (std::holds_alternative<domain::RaiseAction>(possible_action) &&
                    std::get<domain::RaiseAction>(possible_action).amount <= max_raise) {
                    out.push_back(possible_action);
                }
            }
        } else {
            out.push_back(action);
        }
    }
    return out;
}

int train() {
    const learning::MinRegretTrainConfig<RobPoker> config = {
        .num_iterations = 1000,
        .infoset_id_from_hist = [](const RobPoker::History &) { return RobPoker::InfoSetId{}; },
        .action_generator = action_generator,
        .seed = 0,
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
