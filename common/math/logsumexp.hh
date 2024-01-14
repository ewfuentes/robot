
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>

namespace robot::math {
template <typename Accessor = std::identity>
double logsumexp(const auto &terms, const Accessor &accessor = std::identity{}) {
    const auto max_iter = std::max_element(
        terms.begin(), terms.end(),
        [&accessor](const auto &a, const auto &b) { return accessor(a) < accessor(b); });

    if (max_iter == terms.end()) {
        return 0.0;
    }

    const double max_elem = accessor(*max_iter);

    double sum = 0.0;
    for (const auto &item : terms) {
        sum += std::exp(accessor(item) - max_elem);
    }
    return std::log(sum) + max_elem;
}
}  // namespace robot::math
