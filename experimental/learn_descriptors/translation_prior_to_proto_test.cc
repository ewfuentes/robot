#include "experimental/learn_descriptors/translation_prior_to_proto.hh"

#include <iostream>

#include "experimental/learn_descriptors/translation_prior.hh"
#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors {
TEST(TranslationPriorToProtoTest, pack_unpack) {
    // Setup
    Eigen::Matrix3d covariance;
    covariance << 0.94, 0.01, 0.05, 0.01, 0.94, 0.05, 0.05, 0.05, 0.9;
    const learn_descriptors::TranslationPrior in{Eigen::Vector3d{1, 1, 1}, covariance};

    // Action
    proto::TranslationPrior translation_prior_proto;
    pack_into(in, &translation_prior_proto);
    const learn_descriptors::TranslationPrior out = unpack_from(translation_prior_proto);

    // Verification
    EXPECT_TRUE(in.translation.isApprox(out.translation));
    EXPECT_TRUE(in.covariance.isApprox(out.covariance));
}
}  // namespace robot::experimental::learn_descriptors