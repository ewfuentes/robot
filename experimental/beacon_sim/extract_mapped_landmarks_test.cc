
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(ExtractMappedLandmarksTest, extract_landmarks_from_est) {
    // Setup
    constexpr int ROBOT_DIM = 3;
    constexpr int BEACON_DIM = 2;
    const std::array<std::pair<Beacon, double>, 3> beacons_and_cov = {{
        {{
             .id = 100,
             .pos_in_local = {1.0, 2.0},
         },
         3.0},
        {{
             .id = 200,
             .pos_in_local = {3.0, 4.0},
         },
         7.0},
        {{
             .id = 300,
             .pos_in_local = {5.0, 6.0},
         },
         9.0},
    }};

    EkfSlamEstimate est;
    const int dim = ROBOT_DIM + BEACON_DIM * beacons_and_cov.size();
    est.mean = Eigen::VectorXd::Zero(dim);
    est.cov = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i < static_cast<int>(beacons_and_cov.size()); i++) {
        const auto &[beacon, cov_multiplier] = beacons_and_cov.at(i);
        est.beacon_ids.push_back(beacon.id);
        const int beacon_start_idx = ROBOT_DIM + BEACON_DIM * i;
        est.mean(Eigen::seqN(beacon_start_idx, BEACON_DIM)) = beacon.pos_in_local;
        est.cov.block(beacon_start_idx, beacon_start_idx, BEACON_DIM, BEACON_DIM) =
            Eigen::Matrix<double, BEACON_DIM, BEACON_DIM>::Ones() * cov_multiplier;
    }

    // Action
    const MappedLandmarks mapped_landmarks = extract_mapped_landmarks(est);

    // Verification
    EXPECT_EQ(beacons_and_cov.size(), mapped_landmarks.beacon_ids.size());
    EXPECT_EQ(beacons_and_cov.size(), mapped_landmarks.beacon_in_local.size());
    constexpr double TOL = 1e-6;
    for (int i = 0; i < static_cast<int>(beacons_and_cov.size()); i++) {
        const auto &[beacon, cov_multiplier] = beacons_and_cov.at(i);
        EXPECT_EQ(mapped_landmarks.beacon_ids.at(i), beacon.id);
        EXPECT_EQ(mapped_landmarks.beacon_in_local.at(i), beacon.pos_in_local);
        EXPECT_NEAR((mapped_landmarks.cov_in_local.block(i * BEACON_DIM, i * BEACON_DIM, BEACON_DIM,
                                                         BEACON_DIM) -
                     Eigen::Matrix<double, BEACON_DIM, BEACON_DIM>::Ones() * cov_multiplier)
                        .norm(),
                    0.0, TOL);
    }
}
}  // namespace robot::experimental::beacon_sim
