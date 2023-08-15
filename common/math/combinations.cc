
#include "common/math/combinations.hh"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <unordered_map>

#include "common/math/n_choose_k.hh"

namespace robot::math {
std::span<const std::vector<int>> combinations(const int n, const int k) {
    thread_local std::unordered_map<int, std::vector<std::vector<int>>> combinations_cache;
    const int num_combinations = n_choose_k(n, k);
    std::vector<std::vector<int>> &combos = combinations_cache[k];
    if (k == 0) {
        if (combos.empty()) {
            combos.push_back({});
        }
    } else if (k == 1 && n > static_cast<int>(combos.size())) {
        for (int i = combos.size(); i < n; i++) {
            combos.push_back({i});
        }
        return std::span<const std::vector<int>>(combos.begin(), n);
    } else if (n == k && combos.size() == 0) {
        std::vector<int> idxs(n);
        std::iota(idxs.begin(), idxs.end(), 0);
        combos.emplace_back(std::move(idxs));
    }

    if (static_cast<int>(combos.size()) < num_combinations) {
        const auto n_min_one_choose_k = combinations(n - 1, k);
        const auto n_min_one_choose_k_min_one = combinations(n - 1, k - 1);
        assert(combos.size() == n_min_one_choose_k.size());
        for (const auto &idxs : n_min_one_choose_k_min_one) {
            std::vector<int> new_idxs = idxs;
            new_idxs.push_back(n - 1);
            combos.emplace_back((std::move(new_idxs)));
        }
    }

    return std::span<const std::vector<int>>(combos.begin(), num_combinations);
}
}  // namespace robot::math
