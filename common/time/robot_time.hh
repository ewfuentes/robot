
#pragma once

#include <chrono>
#include <cstdint>

namespace robot::time {

// RobotTime represents a std::chrono::time_point which comes from the
// configured clock source.
class RobotTimestamp;
RobotTimestamp current_robot_time();

enum class TimeProvider { STEADY, SIM };
void set_default_time_provider(const TimeProvider &provider);

class RobotTimestamp {
   public:
    using rep = std::int64_t;
    using period = std::nano;
    using duration = std::chrono::duration<rep, period>;

    constexpr RobotTimestamp() : time_since_epoch_(0) {}

    constexpr duration time_since_epoch() const { return time_since_epoch_; }

    constexpr RobotTimestamp &operator+=(const duration &d) {
        time_since_epoch_ += d;
        return *this;
    }

    constexpr RobotTimestamp &operator-=(const duration &d) {
        time_since_epoch_ += -d;
        return *this;
    }

    constexpr RobotTimestamp operator+(const duration &duration) const noexcept {
        RobotTimestamp out = *this;
        out += duration;
        return out;
    }

    static constexpr RobotTimestamp min() noexcept { return RobotTimestamp() + duration::min(); }
    static constexpr RobotTimestamp max() noexcept { return RobotTimestamp() + duration::max(); }

   private:
    duration time_since_epoch_;
};

constexpr RobotTimestamp::duration as_duration(const double time_s) {
    constexpr auto ratio_den = RobotTimestamp::duration::period::den;
    constexpr auto ratio_num = RobotTimestamp::duration::period::num;
    const auto num_ticks =
        static_cast<RobotTimestamp::duration::rep>(time_s * ratio_den / ratio_num);
    return RobotTimestamp::duration(num_ticks);
}

constexpr RobotTimestamp operator-(const RobotTimestamp &a, const RobotTimestamp::duration &b) {
    RobotTimestamp out = a;
    out -= b;
    return out;
}

constexpr RobotTimestamp::duration operator-(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a.time_since_epoch() - b.time_since_epoch();
}

constexpr RobotTimestamp operator+(const RobotTimestamp::duration &a, const RobotTimestamp &b) {
    return b + a;
}

constexpr RobotTimestamp operator-(const RobotTimestamp::duration &a, const RobotTimestamp &b) {
    return b - a;
}

constexpr bool operator==(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a.time_since_epoch() == b.time_since_epoch();
}

constexpr bool operator!=(const RobotTimestamp &a, const RobotTimestamp &b) { return !(a == b); }

constexpr bool operator<(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a.time_since_epoch() < b.time_since_epoch();
}

constexpr bool operator>(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a.time_since_epoch() > b.time_since_epoch();
}

constexpr bool operator>=(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a > b || a == b;
}

constexpr bool operator<=(const RobotTimestamp &a, const RobotTimestamp &b) {
    return a < b || a == b;
}
}  // namespace robot::time
