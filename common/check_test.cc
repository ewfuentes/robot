
#include "common/check.hh"

#include "gtest/gtest.h"

TEST(AssertTest, assert_test) { EXPECT_THROW(ROBOT_CHECK(false), robot::check_failure); }
