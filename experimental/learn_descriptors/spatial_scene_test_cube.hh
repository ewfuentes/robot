#pragma once
#include <iostream>

#include "Eigen/Core"
#include "experimental/learn_descriptors/spatial_scene_test.hh"

namespace robot::experimental::learn_descriptors {
class SpatialSceneTestCube : public SpatialSceneTest {
   public:
    SpatialSceneTestCube(const float cube_size) : SpatialSceneTest() {
        points_.push_back(Eigen::Vector3d(0, 0, 0));
        points_.push_back(Eigen::Vector3d(cube_size, 0, 0));
        points_.push_back(Eigen::Vector3d(cube_size, cube_size, 0));
        points_.push_back(Eigen::Vector3d(0, cube_size, 0));
        points_.push_back(Eigen::Vector3d(0, 0, cube_size));
        points_.push_back(Eigen::Vector3d(cube_size, 0, cube_size));
        points_.push_back(Eigen::Vector3d(cube_size, cube_size, cube_size));
        points_.push_back(Eigen::Vector3d(0, cube_size, cube_size));
    };
};
}  // namespace robot::experimental::learn_descriptors
