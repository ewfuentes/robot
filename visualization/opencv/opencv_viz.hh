#pragma once
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/viz.hpp"

namespace robot::geometry {
struct VizPose {
    const Eigen::Isometry3d world_from_pose;
    const std::optional<std::string> label;
};

struct VizPoint {
    const Eigen::Vector3d t_point_in_world;
    const std::optional<std::string> label;
};
void viz_scene(const std::vector<Eigen::Isometry3d> &world_from_poses,
               const std::vector<Eigen::Vector3d> &points_in_world,
               const cv::viz::Color color_background = cv::viz::Color::black(),
               const bool show_grid = true, const bool show_origin = true,
               const std::string &window_name = "Viz Window", const double text_scale = 0.1);
void viz_scene(const std::vector<VizPose> &world_from_poses,
               const std::vector<VizPoint> &points_in_world,
               const cv::viz::Color color_background = cv::viz::Color::black(),
               const bool show_grid = true, const bool show_origin = true,
               const std::string &window_name = "Viz Window", const double text_scale = 0.1);
}  // namespace robot::geometry