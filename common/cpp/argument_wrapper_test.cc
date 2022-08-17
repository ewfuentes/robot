
#include "common/cpp/argument_wrapper.hh"

#include "gtest/gtest.h"

namespace cpp {
namespace {

// chosen by fair dice roll
// guaranteed to be random
constexpr int RANDOM_NUMBER = 4;
void get_random_number(Out<int> rand_num) { *rand_num = RANDOM_NUMBER; }

void add_one(InOut<int> num) { *num += 1; }
}  // namespace

TEST(ArgumentWrapperTest, make_out) {
    // Setup
    int my_num = 20;

    // Action
    get_random_number(make_out(my_num));

    // Verification
    EXPECT_EQ(my_num, RANDOM_NUMBER);
}

TEST(ArgumentWrapperTest, make_unused) {
    // Action
    get_random_number(make_unused<int>());
    // This test passes if it compiles
}

TEST(ArgumentWrapperTest, make_inout) {
    // Setup
    int num = 42;
    constexpr int EXPECTED = 43;

    // Action
    add_one(make_in_out(num));

    // Verification
    EXPECT_EQ(num, EXPECTED);
}
}  // namespace cpp
