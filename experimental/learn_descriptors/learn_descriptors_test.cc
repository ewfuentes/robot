#include "gtest/gtest.h"
#include "experimental/learn_descriptors/learn_descriptors.hh"

TEST(LearnDescriptorsTest, hello_world) {
    robot::experimental::learn_descriptors::hello_world("sup");
}