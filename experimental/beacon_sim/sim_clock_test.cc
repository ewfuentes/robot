
#include "experimental/beacon_sim/sim_clock.hh"

#include <chrono>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
  TEST(SimClockTest, is_clock) { static_assert(std::chrono::is_clock<SimClock>()); }
  TEST(SimClockTest, advances_as_expected) {
    constexpr SimClock::duration DT = 100ms;
    const auto t0 = SimClock::now();

    SimClock::advance(DT);
    const auto t1 = SimClock::now();
    SimClock::advance(DT);

    EXPECT_EQ(t1 - t0, DT);
  }
}  // namespace robot::experimental::beacon_sim
