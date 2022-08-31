
#include "common/liegroups/se3.hh"

#include <numbers>

#include "gtest/gtest.h"

namespace robot::liegroups {
TEST(SE3Test, equal_log_translation_yields_equal_arclength) {
    // Setup
    constexpr double LOG_TRANSLATION_Y = 10.0;
    constexpr double ROTATION = std::numbers::pi;
    const SE3 b_from_a = SE3::exp({0.0, LOG_TRANSLATION_Y, 0.0, 0.0, 0.0, 0.0});
    const SE3 b_from_c = SE3::exp({0.0, LOG_TRANSLATION_Y, 0.0, ROTATION, 0.0, 0.0});

    // Action + Verification
    EXPECT_EQ(b_from_a.arclength(), b_from_c.arclength());
}
}  // namespace robot::liegroups
