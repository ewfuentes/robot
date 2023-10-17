
#include "assert/assert.hpp"

#define CHECK(expr, ...) ASSERT_INVOKE(expr, false, true, "CHECK", verification, , __VA_ARGS__)

namespace robot {
using check_failure = libassert::verification_failure;
}
