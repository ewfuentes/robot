#pragma once
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/viz.hpp"

namespace robot::geometry::opencv_viz {
void viz_scene(const std::vector<Eigen::Isometry3d> &poses,
               const std::vector<Eigen::Vector3d> &points, const bool show_grid = true);
}