
#pragma once

#include <chrono>
#include <cstdint>

namespace robot::experimental::beacon_sim {

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

    static constexpr RobotTimestamp min() noexcept;
    static constexpr RobotTimestamp max() noexcept;

   private:
    duration time_since_epoch_;
};

constexpr RobotTimestamp operator+(const RobotTimestamp &a, const RobotTimestamp::duration &b) {
    RobotTimestamp out = a;
    out += b;
    return out;
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
};

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
}  // namespace robot::experimental::beacon_sim
