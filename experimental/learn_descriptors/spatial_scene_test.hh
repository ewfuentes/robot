#pragma once
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "gtsam/geometry/PinholeCamera.h"
#include "opencv2/opencv.hpp"

class ProjectionHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] >= 0 && pixel[0] < img_width && pixel[1] >= 0 && pixel[1] < img_height;
    }
};
namespace robot::experimental::learn_descriptors {
class SpatialSceneTest {
   public:
    SpatialSceneTest() = default;
    SpatialSceneTest(std::vector<Eigen::Vector3d> points) : points_(points){};
    const std::vector<Eigen::Vector3d> &get_points() { return points_; };
    template <typename T>
    std::vector<Eigen::Vector2d> get_projected_pixels(const gtsam::PinholeCamera<T> &camera,
                                                      const cv::Size &img_size) {
        std::vector<Eigen::Vector2d> pixels;
        for (const Eigen::Vector3d &pt : points_) {
            Eigen::Vector2d pixel = camera.project(pt);
            if (ProjectionHelper::pixel_in_range(pixel, img_size.width, img_size.height)) {
                pixels.push_back(pixel);
            }
        }
        return pixels;
    };
    template <typename T>
    const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> get_corresponding_pixels(
        const std::vector<std::pair<gtsam::PinholeCamera<T>, cv::Size>> &cam_and_img_size) {
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> corresponding_pixels;
        for (const Eigen::Vector3d &pt : points_) {
            Eigen::Vector2d pixel_cam0 = cam_and_img_size[0].first.project(pt);
            Eigen::Vector2d pixel_cam1 = cam_and_img_size[1].first.project(pt);
            if (ProjectionHelper::pixel_in_range(pixel_cam0, cam_and_img_size[0].second.width,
                                                 cam_and_img_size[0].second.height) &&
                ProjectionHelper::pixel_in_range(pixel_cam1, cam_and_img_size[1].second.width,
                                                 cam_and_img_size[1].second.height)) {
                corresponding_pixels.emplace_back(pixel_cam0, pixel_cam1);
            }
        }
        return corresponding_pixels;
    };

   protected:
    std::vector<Eigen::Vector3d> points_;
};
}  // namespace robot::experimental::learn_descriptors