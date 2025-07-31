#pragma once

#include "Eigen/Core"

namespace robot::experimental::learn_descriptors {
struct TranslationPrior {
    Eigen::Vector3d translation;
    Eigen::Matrix3d covariance;
};
}  // namespace robot::experimental::learn_descriptors