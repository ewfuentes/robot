
#include "common/matplotlib.hh"

#include "gtest/gtest.h"

namespace robot {

TEST(MatplotlibTest, simple_plot) {
    EXPECT_NO_THROW(plot(
        {{.x = std::vector{0.0, 1.0, 2.0}, .y = std::vector{10.0, 20.0, 15.0}, .label = "label"}},
        false));
}
}  // namespace robot
