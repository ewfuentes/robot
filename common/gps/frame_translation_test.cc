#include "common/gps/frame_translation.hh"

#include "Eigen/Core"
#include "common/check.hh"
#include "gtest/gtest.h"

namespace robot::gps {
TEST(frame_translation_test, lla_from_ecef) {
    const Eigen::Vector3d lla_easthampton_ma(42.2608, -72.6634, 51.82);
    const Eigen::Vector3d ecef_easthampton_ma = ecef_from_lla(lla_easthampton_ma);

    const Eigen::Vector3d ecef_expected_easthampton_ma(1408753.87, -4512832.60, 4267122.34);

    constexpr double tol = 1e-9;
    ROBOT_CHECK(ecef_easthampton_ma.isApprox(ecef_expected_easthampton_ma, tol),
                ecef_easthampton_ma, ecef_expected_easthampton_ma);
}

TEST(frame_translation_test, ecef_from_lla) {
    const Eigen::Vector3d ecef_palace_versailles(4205987.87, 155694.68, 4776399.15);
    const Eigen::Vector3d lla_palace_versailles = lla_from_ecef(ecef_palace_versailles);

    const Eigen::Vector3d expected_lla_palace_versailles(48.8049440, 2.1199720, 131.985);

    constexpr double tol = 1e-4;
    ROBOT_CHECK(lla_palace_versailles.isApprox(expected_lla_palace_versailles, tol),
                lla_palace_versailles, expected_lla_palace_versailles);
}
}  // namespace robot::gps