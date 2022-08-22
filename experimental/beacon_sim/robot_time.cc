
#include "experimental/beacon_sim/robot_time.hh"

#include "experimental/beacon_sim/sim_clock.hh"

namespace robot::experimental::beacon_sim {
namespace {
static TimeProvider time_provider = TimeProvider::STEADY;
}
void set_default_time_provider(const TimeProvider &provider) {
    time_provider = provider;
    if (provider == TimeProvider::SIM) {
        SimClock::reset();
    }
}

RobotTimestamp current_robot_time() {
    if (time_provider == TimeProvider::STEADY) {
        return std::chrono::steady_clock::now().time_since_epoch() + RobotTimestamp();
    } else if (time_provider == TimeProvider::SIM) {
        return SimClock::now().time_since_epoch() + RobotTimestamp();
    }
    return RobotTimestamp();
}

constexpr RobotTimestamp RobotTimestamp::min() noexcept {
    return duration::min() + RobotTimestamp();
}
constexpr RobotTimestamp RobotTimestamp::max() noexcept {
    return duration::max() + RobotTimestamp();
}
}  // namespace robot::experimental::beacon_sim
