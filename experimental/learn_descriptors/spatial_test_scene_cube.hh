#pragma once
#include <iostream>

#include "Eigen/Core"
#include "experimental/learn_descriptors/spatial_test_scene.hh"

namespace robot::experimental::learn_descriptors {
class SpatialTestSceneCube : public SpatialTestScene {
   public:
    SpatialTestSceneCube(const float cube_size) : SpatialTestScene() {
        points_.push_back(Eigen::Vector3d(-cube_size / 2, -cube_size / 2, -cube_size / 2));
        points_.push_back(Eigen::Vector3d(-cube_size / 2, cube_size / 2, -cube_size / 2));
        points_.push_back(Eigen::Vector3d(cube_size / 2, -cube_size / 2, -cube_size / 2));
        points_.push_back(Eigen::Vector3d(cube_size / 2, cube_size / 2, -cube_size / 2));
        points_.push_back(Eigen::Vector3d(-cube_size / 2, -cube_size / 2, cube_size / 2));
        points_.push_back(Eigen::Vector3d(-cube_size / 2, cube_size / 2, cube_size / 2));
        points_.push_back(Eigen::Vector3d(cube_size / 2, -cube_size / 2, cube_size / 2));
        points_.push_back(Eigen::Vector3d(cube_size / 2, cube_size / 2, cube_size / 2));
    };
};
}  // namespace robot::experimental::learn_descriptors
