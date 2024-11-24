
#include "common/matplotlib.hh"

#include "gtest/gtest.h"

namespace robot {

TEST(MatplotlibTest, simple_plot) { EXPECT_NO_THROW(plot({0.0, 1.0, 2.0}, {10.0, 20.0, 15.0})); }
}  // namespace robot
