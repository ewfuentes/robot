
#pragma once

#include <random>
#include <vector>

#include "common/argument_wrapper.hh"

namespace robot::math {
std::vector<int> sample_without_replacement(const std::vector<double> &p, const int num_samples,
                                            const bool is_log_p, InOut<std::mt19937> gen);
}
