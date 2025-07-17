#pragma once

#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/PinholeCamera.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class ProjectionHelper {
   public:
    static constexpr size_t img_width_default = 640;
    static constexpr size_t img_height_default = 480;
    static constexpr double fx_default = 500.0;
    static constexpr double fy_default = fx_default;
    static constexpr double cx_default = img_width_default / 2.0;
    static constexpr double cy_default = img_height_default / 2.0;
    static const gtsam::Cal3_S2::shared_ptr &K_default() {
        static gtsam::Cal3_S2::shared_ptr K(
            new gtsam::Cal3_S2(fx_default, fy_default, 0.0, cx_default, cy_default));
        return K;
    }
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] >= 0 && pixel[0] < img_width && pixel[1] >= 0 && pixel[1] < img_height;
    }
};
class SpatialTestScene {
   public:
    struct Noise {
        Noise(const std::vector<double> &sigmas, std::optional<unsigned int> seed = std::nullopt)
            : sigmas_(sigmas) {
            if (seed) {
                gen_.seed(*seed);
            } else {
                std::random_device rd;
                gen_.seed(rd());
            }

            for (double sigma : sigmas_) {
                dists_.emplace_back(0.0, sigma);
            }
        }

        static Noise constrained(size_t dim = 6) { return Noise(std::vector<double>(dim, 0.0)); }

        Eigen::VectorXd sample() {
            Eigen::VectorXd result(sigmas_.size());
            for (size_t i = 0; i < sigmas_.size(); ++i) {
                result(i) = dists_[i](gen_);
            }
            return result;
        }

        size_t dim() const { return sigmas_.size(); }

       private:
        std::vector<double> sigmas_;
        std::vector<std::normal_distribution<double>> dists_;
        std::mt19937 gen_;
    };
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
        const std::vector<std::pair<gtsam::PinholeCamera<T>, cv::Size>>
            &cam_and_img_size);  // make this parameter of size 2

    void add_point(Eigen::Vector3d point, std::optional<Noise> point_noise = std::nullopt);
    void add_points(std::vector<Eigen::Vector3d> points,
                    std::optional<Noise> point_noise = std::nullopt);
    void add_camera(gtsam::PinholeCamera<gtsam::Cal3_S2> camera,
                    std::optional<Noise> pose_noise = std::nullopt);
    void add_cameras(std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras,
                     std::optional<Noise> pose_noise = std::nullopt);
    void add_rand_cameras_face_origin(int num_cameras, double min_radius_origin,
                                      double max_radius_origin,
                                      std::optional<Noise> pose_noise = std::nullopt,
                                      const gtsam::Cal3_S2 &K = *ProjectionHelper::K_default());

    const std::vector<Eigen::Vector3d> &points() { return points_; };
    const std::vector<Eigen::Vector3d> &points_groundtruth() { return points_groundtruth_; };
    const std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras() { return cameras_; };
    const std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> &cameras_groundtruth() {
        return cameras_groundtruth_;
    };

   protected:
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Vector3d> points_groundtruth_;
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras_;
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras_groundtruth_;
};
}  // namespace robot::experimental::learn_descriptors