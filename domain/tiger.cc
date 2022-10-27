
#include "domain/tiger.hh"

namespace robot::domain {
Tiger::Tiger(const TigerConfig &config, const std::size_t seed)
    : config_{config}, gen_{seed}, is_done_(false) {
    std::uniform_int_distribution<> distrib(0, 1);
    is_tiger_left_ = distrib(gen_);
}

Tiger::Result Tiger::step(const Tiger::Action &action) {
    if (is_done_) {
        return Tiger::Result{
            .reward = 0.0, .observation = Tiger::Observation::INVALID, .is_done = true};
    } else if (action == Tiger::Action::LISTEN) {
        std::uniform_real_distribution<> distrib(0.0, 1.0);
        const bool is_consistent = distrib(gen_) < config_.consistent_observation_probability;

        return Tiger::Result{
            .reward = config_.listening_reward,
            .observation = is_consistent ? (is_tiger_left_ ? Tiger::Observation::GROWL_LEFT
                                                           : Tiger::Observation::GROWL_RIGHT)
                                         : (is_tiger_left_ ? Tiger::Observation::GROWL_RIGHT
                                                           : Tiger::Observation::GROWL_RIGHT),
            .is_done = false,
        };
    }

    is_done_ = true;
    const bool is_tiger = (is_tiger_left_ && action == Tiger::Action::OPEN_LEFT) ||
                          (!is_tiger_left_ && action == Tiger::Action::OPEN_RIGHT);
    return Tiger::Result {
        .reward = is_tiger ? config_.tiger_reward : config_.treasure_reward,
        .observation = is_tiger ? Tiger::Observation::TIGER : Tiger::Observation::TREASURE,
        .is_done = is_done_,
    };
}
}  // namespace robot::domain
