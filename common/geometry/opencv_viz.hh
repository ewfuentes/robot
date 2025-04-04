#pragma once
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/viz.hpp"

namespace robot::geometry {
void viz_scene(const std::vector<Eigen::Isometry3d> &poses_world,
               const std::vector<Eigen::Vector3d> &points_world, const bool show_grid = true,
               const bool show_origin = true, const std::string &window_name = "Viz Window");
}