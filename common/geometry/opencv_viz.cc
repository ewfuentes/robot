#include "common/geometry/opencv_viz.hh"

#include <iostream>

#include "common/geometry/translate_types.hh"

namespace robot::geometry {
cv::Vec3d rotation_matrix_to_axis_angle(const cv::Matx33d &R) {
    // Ensure R is a valid rotation matrix
    CV_Assert(cv::determinant(R) > 0.999 && cv::determinant(R) < 1.001);

    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double theta = std::acos((trace - 1) / 2.0);

    if (std::abs(theta) < 1e-6) {
        return cv::Vec3d(0, 0, 0);
    }

    double sinTheta = std::sin(theta);
    cv::Vec3d axis((R(2, 1) - R(1, 2)) / (2.0 * sinTheta), (R(0, 2) - R(2, 0)) / (2.0 * sinTheta),
                   (R(1, 0) - R(0, 1)) / (2.0 * sinTheta));

    return axis * theta;
}

void viz_scene(const std::vector<Eigen::Isometry3d> &poses_world,
               const std::vector<Eigen::Vector3d> &points_world, const bool show_grid,
               const bool show_origin, const std::string &window_name) {
    cv::viz::Viz3d window(window_name);

    constexpr double pose_size = .5;
    for (unsigned int i = 0; i < poses_world.size(); i++) {
        cv::Affine3d cv_pose(eigen_mat_to_cv(Eigen::Matrix4d(poses_world[i].matrix())));
        window.showWidget("rigid_transform_" + std::to_string(i),
                          cv::viz::WCoordinateSystem(pose_size), cv_pose);
    }
    constexpr double point_radius = 0.08;
    constexpr int sphere_res = 10;
    const cv::viz::Color point_color = cv::viz::Color::celestial_blue();
    for (unsigned int i = 0; i < points_world.size(); i++) {
        const Eigen::Vector3d &point = points_world[i];
        window.showWidget("point_" + std::to_string(i),
                          cv::viz::WSphere(cv::Point3d(point[0], point[1], point[2]), point_radius,
                                           sphere_res, point_color));
    }

    if (show_origin) {
        window.showWidget("world_frame", cv::viz::WCoordinateSystem());
    }

    if (show_grid) {
        constexpr unsigned int num_cells = 6;
        constexpr double units = 1.0;
        const cv::viz::Color color = cv::viz::Color::gray();
        const cv::Vec2i cells = cv::Vec2i::all(num_cells);
        const cv::Vec2d cell_sizes = cv::Vec2d::all(units);
        window.showWidget("xy_grid", cv::viz::WGrid(cells, cell_sizes, color));
        window.showWidget("yz_grid",
                          cv::viz::WGrid(cv::Point3d(0., 0., 0.), cv::Point3d(1, 0, 0),
                                         cv::Point3d(0, 0, 1), cells, cell_sizes, color));
        window.showWidget("xz_grid",
                          cv::viz::WGrid(cv::Point3d(0., 0., 0.), cv::Point3d(0, 1, 0),
                                         cv::Point3d(0, 0, 1), cells, cell_sizes, color));
    }

    window.spin();
}
}  // namespace robot::geometry
