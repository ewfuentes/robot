
#include <cstdint>

namespace robot::math {
constexpr std::uint64_t n_choose_k(const int n, const int k) {
    std::uint64_t out = 1;
    for (int i = 1; i <= k; i++) {
        out = (out * (n - k + i)) / i;
    }
    return out;
}
}  // namespace robot::math
