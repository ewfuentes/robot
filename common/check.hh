
#include "assert/assert.hpp"

#define ROBOT_CHECK(expr, ...) \
    ASSERT_INVOKE(expr, false, true, "ROBOT_CHECK", verification, , __VA_ARGS__)

namespace robot {
using check_failure = libassert::verification_failure;
}
