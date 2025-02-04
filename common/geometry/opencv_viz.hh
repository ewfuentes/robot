#pragma once
#include <vector>
#include "opencv2/viz.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"

namespace robot::geometry::opencv_viz {
void viz_scene(const std::vector<Eigen::Isometry3d> &poses, const std::vector<Eigen::Vector3d> &points);
}