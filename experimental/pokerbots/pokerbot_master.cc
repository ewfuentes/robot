
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
            {domain::RaiseAction{5}, "Raise5"},
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

int train() {
    const learning::MinRegretTrainConfig<RobPoker> config = {
        .num_iterations = 1000,
        .infoset_id_from_hist = [](const RobPoker::History &) { return RobPoker::InfoSetId{}; },
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
