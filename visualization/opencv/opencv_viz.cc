#include "visualization/opencv/opencv_viz.hh"

#include <iostream>

#include "common/geometry/translate_types.hh"
#include "opencv2/viz.hpp"

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

void viz_scene(const std::vector<Eigen::Isometry3d> &world_from_poses,
               const std::vector<Eigen::Vector3d> &points_in_world,
               const cv::viz ::Color color_background, const bool show_grid, const bool show_origin,
               const std::string &window_name, const double text_scale) {
    std::vector<VizPose> viz_poses;
    std::vector<VizPoint> viz_points;
    for (const Eigen::Isometry3d &world_from_pose : world_from_poses) {
        viz_poses.emplace_back(world_from_pose);
    }
    for (const Eigen::Vector3d &t_point_in_world : points_in_world) {
        viz_points.emplace_back(t_point_in_world);
    }
    viz_scene(viz_poses, viz_points, color_background, show_grid, show_origin, window_name,
              text_scale);
}

void viz_scene(const std::vector<VizPose> &world_from_poses,
               const std::vector<VizPoint> &points_in_world, const cv::viz ::Color color_background,
               const bool show_grid, const bool show_origin, const std::string &window_name,
               const double text_scale) {
    cv::viz::Viz3d window(window_name);

    window.setBackgroundColor(color_background);

    constexpr bool ALWAYS_FACE_CAMERA = true;
    constexpr double pose_size = .5;
    for (unsigned int i = 0; i < world_from_poses.size(); i++) {
        cv::Affine3d cv_pose(
            eigen_mat_to_cv(Eigen::Matrix4d(world_from_poses[i].world_from_pose.matrix())));
        window.showWidget("rigid_transform_" + std::to_string(i),
                          cv::viz::WCoordinateSystem(pose_size), cv_pose);
        if (world_from_poses[i].label) {
            window.showWidget(
                "text_pose_" + std::to_string(i),
                cv::viz::WText3D(
                    *(world_from_poses[i].label),
                    cv::Point3d(eigen_vec_to_cv(
                        world_from_poses[i].world_from_pose.translation() +
                        Eigen::Vector3d(0, 0,
                                        0.001))),  // small offset is for occasional rendering bug
                    text_scale, ALWAYS_FACE_CAMERA));
        }
    }
    constexpr double point_radius = 0.08;
    constexpr int sphere_res = 10;
    const cv::viz::Color point_color = cv::viz::Color::celestial_blue();
    for (unsigned int i = 0; i < points_in_world.size(); i++) {
        const Eigen::Vector3d &point = points_in_world[i].t_point_in_world;
        std::cout << "pt to viz: " << point << std::endl;
        window.showWidget("point_" + std::to_string(i),
                          cv::viz::WSphere(cv::Point3d(point[0], point[1], point[2]), point_radius,
                                           sphere_res, point_color));
        if (points_in_world[i].label) {
            window.showWidget(
                "text_point_" + std::to_string(i),
                cv::viz::WText3D(
                    *(points_in_world[i].label),
                    cv::Point3d(eigen_vec_to_cv(
                        points_in_world[i].t_point_in_world +
                        Eigen::Vector3d(0, 0,
                                        0.001))),  // small offset is for occasional rendering bug
                    text_scale, ALWAYS_FACE_CAMERA));
        }
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

    // std::cout << "viz heartbeat" << std::endl;
    // window.setViewerPose(cv::Affine3d(
    //     eigen_mat_to_cv(Eigen::Matrix4d(world_from_poses[0].world_from_pose.matrix()))));
    // window.setViewerPose(cv::Affine3d::Identity());
    // std::cout << "success" << std::endl;

    window.spin();
}
}  // namespace robot::geometry
