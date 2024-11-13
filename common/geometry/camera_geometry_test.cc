#include "common/geometry/camera_geometry.hh"

#include "gtest/gtest.h"

#include <iostream>

namespace robot::geometry {
TEST(CameraGeometryTest, pixel_and_spatial) {
    const double fx = 500.0;
    const double fy = fx;
    const double cx = 320;
    const double cy = 240;

    const Eigen::Vector3d point(1,0.5,2);

    const Eigen::Vector2d pixel = project_point_to_camera_pixel(point, fx, fy, cx, cy);
    const Eigen::Vector3d recovered_point = project_camera_pixel_to_3D(pixel, point[2], fx, fy, cx, cy);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(recovered_point[0], point[0], TOL);
    EXPECT_NEAR(recovered_point[1], point[1], TOL);
    EXPECT_NEAR(recovered_point[2], point[2], TOL);
}
}