
#include "common/math/sample_without_replacement.hh"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>

#include "common/check.hh"

namespace robot::math {
std::vector<int> sample_without_replacement(const std::vector<double> &weights,
                                            const int num_samples, const bool is_log_weights,
                                            InOut<std::mt19937> gen) {
    if (num_samples >= static_cast<int>(weights.size())) {
        std::vector<int> out(weights.size());
        std::iota(out.begin(), out.end(), 0);
        return out;
    }

    std::vector<std::tuple<int, double>> idx_and_sample;
    const double max_elem = *std::max_element(weights.begin(), weights.end());
    std::uniform_real_distribution<> dist;
    for (int i = 0; i < static_cast<int>(weights.size()); i++) {
        const double weight = is_log_weights ? std::exp(weights.at(i) - max_elem) : weights.at(i);
        const double rand = std::pow(dist(*gen), 1 / weight);
        idx_and_sample.push_back(std::make_tuple(i, rand));
    }

    const auto middle = idx_and_sample.begin() + num_samples;
    std::partial_sort(idx_and_sample.begin(), middle, idx_and_sample.end(),
                      [](const auto &a, const auto &b) {
                          // We want the `num_samples` largest entries, so we return if a  > b
                          return std::get<double>(a) > std::get<double>(b);
                      });

    std::vector<int> out;
    out.reserve(num_samples);
    std::transform(idx_and_sample.begin(), middle, std::back_inserter(out),
                   [](const auto &a) { return std::get<int>(a); });

    return out;
}
}  // namespace robot::math
