
#pragma once

#include <chrono>
#include <cstdint>
#include <ratio>

namespace robot::time {
using namespace std::literals::chrono_literals;
class SimClock {
   public:
    using rep = std::uint64_t;
    using period = std::nano;
    using duration = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<SimClock>;
    static constexpr bool is_steady = false;

    static time_point now() { return now_; }

    static time_point advance(const duration dt) {
        now_ += dt;
        return now();
    }

    static time_point reset() {
        now_ = {};
        return now();
    }

   private:
    // Default constructed to zero
    static time_point now_;
};
}  // namespace robot::time
