#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>

#include "Eigen/Dense"
#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/backend.hh"
#include "experimental/learn_descriptors/feature_manager.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/Values.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class StructureFromMotion {
   public:
    static const Eigen::Isometry3d T_symlake_boat_cam;
    static const gtsam::Pose3 default_initial_pose;
    /**
     * @param D is vector (5x1) of the distortion coefficients (k1, k2, p1, p2, k3)
     */
    StructureFromMotion(Frontend::ExtractorType frontend_extractor, gtsam::Cal3_S2 K,
                        Eigen::Matrix<double, 5, 1> D,
                        gtsam::Pose3 initial_pose = default_initial_pose,
                        Frontend::MatcherType frontend_matcher = Frontend::MatcherType::KNN);
    ~StructureFromMotion(){};

    void set_initial_pose(gtsam::Pose3 initial_pose);
    void add_image(const cv::Mat &img, const gtsam::Pose3 &T_world_cam);
    void solve_structure() { backend_.solve_graph(); };
    void solve_structure(
        const int num_steps,
        std::optional<Backend::graph_step_debug_func> inter_debug_func = std::nullopt);
    const gtsam::Values &get_structure_result() { return backend_.get_result(); };
    using match_function = std::function<void(std::vector<cv::DMatch> &)>;
    std::vector<cv::DMatch> get_matches(
        const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
        std::optional<match_function> post_process_func = std::nullopt);
    void graph_values(const gtsam::Values &values, const std::string &window_name = "graph values");

    Frontend get_frontend() { return frontend_; };
    Backend get_backend() { return backend_; }
    size_t get_num_images_added() { return img_keypoints_and_descriptors_.size(); };
    size_t get_landmark_count() { return landmark_count_; };
    const std::vector<std::vector<cv::DMatch>> get_matches() { return matches_; };

   private:
    std::shared_ptr<FeatureManager> feature_manager_;

    gtsam::Pose3 initial_pose_;
    std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> img_keypoints_and_descriptors_;
    std::vector<std::vector<cv::DMatch>> matches_;
    std::vector<std::vector<Backend::Landmark>> landmarks_;
    std::vector<std::unordered_map<cv::KeyPoint, Backend::Landmark>> keypoint_to_landmarks_;
    size_t landmark_count_ = 0;

    cv::Mat K_;
    cv::Mat D_;

    Frontend frontend_;
    Backend backend_;
};
}  // namespace robot::experimental::learn_descriptors