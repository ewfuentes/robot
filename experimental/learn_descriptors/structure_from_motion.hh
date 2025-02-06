#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "Eigen/Core"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/GeneralSFMFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"
#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"

using json = nlohmann::json;

namespace robot::experimental::learn_descriptors {
class Frontend {
   public:
    enum class ExtractorType { SIFT, ORB };
    enum class MatcherType { BRUTE_FORCE, KNN, FLANN };

    Frontend(){};
    Frontend(ExtractorType frontend_extractor, MatcherType frontend_matcher);
    ~Frontend(){};

    const ExtractorType &get_extractor_type() const { return extractor_type_; };
    const MatcherType &get_matcher_type() const { return matcher_type_; };

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> get_keypoints_and_descriptors(
        const cv::Mat &img) const;
    std::vector<cv::DMatch> get_matches(const cv::Mat &descriptors1,
                                        const cv::Mat &descriptors2) const;
    static void threshold_matches(std::vector<cv::DMatch> &matches, float dist_threshhold);
    static void enforce_bijective_matches(std::vector<cv::DMatch> &matches);
    static void draw_keypoints(const cv::Mat &img, std::vector<cv::KeyPoint> keypoints,
                               cv::Mat img_keypoints_out) {
        cv::drawKeypoints(img, keypoints, img_keypoints_out, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    static void draw_matches(const cv::Mat &img1, std::vector<cv::KeyPoint> keypoints1,
                             const cv::Mat &img2, std::vector<cv::KeyPoint> keypoints2,
                             std::vector<cv::DMatch> matches, cv::Mat img_matches_out) {
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches_out);
    }

   private:
    bool get_brute_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                           std::vector<cv::DMatch> &matches_out) const;
    bool get_KNN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                         std::vector<cv::DMatch> &matches_out) const;
    bool get_FLANN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                           std::vector<cv::DMatch> &matches_out) const;
    ExtractorType extractor_type_;
    MatcherType matcher_type_;

    cv::Ptr<cv::Feature2D> feature_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
};
class Backend {
   public:
    static constexpr char pose_symbol_char = 'x';
    static constexpr char pose_rot_symbol_char = 'r';
    static constexpr char pose_bearing_symbol_char = 'b';
    static constexpr char landmark_symbol_char = 'l';
    static constexpr char camera_symbol_char = 'k';

    struct Landmark {
        Landmark(const gtsam::Symbol &lmk_factor_symbol, const gtsam::Symbol &cam_pose_symbol,
                 const gtsam::Point2 &projection,
                 const gtsam::Point3 &initial_guess = gtsam::Point3::Identity())
            : lmk_factor_symbol(lmk_factor_symbol),
              cam_pose_symbol(cam_pose_symbol),
              projection(projection),
              initial_guess(initial_guess){};
        const gtsam::Symbol lmk_factor_symbol;
        const gtsam::Symbol cam_pose_symbol;
        const gtsam::Point2 projection;
        const gtsam::Point3 initial_guess;
    };

    Backend();
    Backend(gtsam::Cal3_S2 K);
    ~Backend(){};

    void add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Pose3 &value);

    template <typename T>
    void add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2,
                            const T &value, const gtsam::SharedNoiseModel &model);


    void add_landmarks(const std::vector<Landmark> &landmarks);
    void add_landmark(const Landmark &landmark);
    gtsam::Pose3 estimate_c0_c1(const std::vector<cv::KeyPoint> &kpts1,
                               const std::vector<cv::KeyPoint> &kpts2,
                               const std::vector<cv::DMatch> &matches, const gtsam::Cal3_S2 &K);

    void solve_graph();    

    const gtsam::Values &get_current_initial_values() const { return initial_estimate_; };
    const gtsam::Values &get_result() const { return result_; };
    const gtsam::Cal3_S2 &get_K() const { return K_; };

   private:
    gtsam::Cal3_S2 K_;

    gtsam::Values initial_estimate_;
    gtsam::Values result_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::noiseModel::Diagonal::shared_ptr pose_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.1));
    gtsam::noiseModel::Isotropic::shared_ptr rotation_noise_ = 
        gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    gtsam::noiseModel::Isotropic::shared_ptr pose_bearing_noise_ = 
        gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
};
class StructureFromMotion {
   public:
    static const Eigen::Affine3d T_symlake_boat_cam;
    static const gtsam::Pose3 default_initial_pose;
    StructureFromMotion(Frontend::ExtractorType frontend_extractor, gtsam::Cal3_S2 K,
                        gtsam::Pose3 initial_pose = default_initial_pose,
                        Frontend::MatcherType frontend_matcher = Frontend::MatcherType::KNN);
    ~StructureFromMotion(){};

    void set_initial_pose(gtsam::Pose3 initial_pose);
    void add_image(const cv::Mat &img);
    void solve_structure() { backend_.solve_graph(); };
    const gtsam::Values &get_structure_result() { return backend_.get_result(); };

    Frontend get_frontend() { return frontend_; };
    Backend get_backend() { return backend_; }
    size_t get_num_images_added() { return img_keypoints_and_descriptors_.size(); };
    size_t get_landmark_count() { return landmark_count_; };

   private:
    std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> img_keypoints_and_descriptors_;
    std::vector<std::vector<cv::DMatch>> matches_;
    std::vector<std::vector<Backend::Landmark>> landmarks_;
    size_t landmark_count_ = 0;

    Frontend frontend_;
    Backend backend_;
};
// class SFM_Logger {
//     public:
//         // static const json gtsam_pose3_to_json(const gtsam::Pose3 &pose);
//         static const json eigen_mat_to_json(const Eigen::Matrix &mat);
//         static const json gtsam_pose3_to_json(const gtsam::Pose3 &pose);
//         static const void values_to_json(const gtsam::Values &values, const std::filesystem::path
//         &file);
// };
}  // namespace robot::experimental::learn_descriptors