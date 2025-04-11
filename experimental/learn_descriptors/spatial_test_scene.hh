#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/PinholeCamera.h"
#include "opencv2/opencv.hpp"

class ProjectionHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] >= 0 && pixel[0] < img_width && pixel[1] >= 0 && pixel[1] < img_height;
    }
};
namespace robot::experimental::learn_descriptors {
class SpatialTestScene {
   public:
    struct ProjectedPoint {
        ProjectedPoint(const size_t pt_idx, const Eigen::Vector2d &pixel)
            : pt_idx(pt_idx), pixel(pixel){};
        size_t pt_idx;
        Eigen::Vector2d pixel;
    };
    SpatialTestScene() = default;
    SpatialTestScene(std::vector<Eigen::Vector3d> points) : points_(points){};
    template <typename T>
    std::vector<ProjectedPoint> get_projected_pixels(const gtsam::PinholeCamera<T> &camera,
                                                     const cv::Size &img_size);
    template <typename T>
    std::vector<std::pair<ProjectedPoint, ProjectedPoint>> get_corresponding_pixels(
        const std::pair<gtsam::PinholeCamera<T>, cv::Size> (&cam_and_img_size)[2]);

    void add_point(Eigen::Vector3d point);
    void add_points(std::vector<Eigen::Vector3d> points);
    void add_camera(gtsam::PinholeCamera<gtsam::Cal3_S2> camera);
    void add_cameras(std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras);
    void add_rand_cameras_face_origin(int num_cameras, double min_radius_origin,
                                      double max_radius_origin, const gtsam::Cal3_S2 &K);

    std::vector<Eigen::Vector3d> &get_points() { return points_; };
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> &get_cameras() { return cameras_; };

   protected:
    std::vector<Eigen::Vector3d> points_;
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras_;
};
}  // namespace robot::experimental::learn_descriptors