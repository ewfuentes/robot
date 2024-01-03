
#include "common/math/sample_without_replacement.hh"

#include <algorithm>
#include <functional>
#include <numeric>

#include "common/check.hh"

namespace robot::math {
absl::flat_hash_set<int> reservoir_sample_without_replacement(std::vector<double> weights,
                                                    const int num_samples,
                                                    const bool is_log_weights,
                                                    InOut<std::mt19937> gen) {
    CHECK(num_samples <= weights.size());
    absl::flat_hash_set<int> out;

    if (num_samples == static_cast<int>(weights.size())) {
        for (int i = 0; i < num_samples; i++) {
            out.insert(i);
        }
        return out;
    }

    std::vector<double> weights_copy = weights;
    if (is_log_weights) {
        const double max_elem = *std::max_element(weights.begin(), weights.end());
        for (double &weight : weights_copy) {
            weight = std::exp(weight - max_elem);
        }
    }

    for (int i = 0; i < num_samples; i++) {
        // Sample
        std::discrete_distribution<> dist(weights_copy.begin(), weights_copy.end());
        const int sampled_idx = dist(*gen);
        out.insert(sampled_idx);

        // Remove the selected index
        weights_copy.at(sampled_idx) = 0.0;
    }

    return out;
}
}  // namespace robot::math
