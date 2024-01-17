
#pragma once

#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "common/argument_wrapper.hh"

namespace robot::math {
absl::flat_hash_set<int> reservoir_sample_without_replacement(const std::vector<double> &p,
                                                              const int num_samples,
                                                              const bool is_log_p,
                                                              InOut<std::mt19937> gen);
}
