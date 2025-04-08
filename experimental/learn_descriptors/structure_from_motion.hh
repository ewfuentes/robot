#pragma once

#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "Eigen/Core"
#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/feature_manager.hh"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/GPSFactor.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "opencv2/opencv.hpp"

// namespace std {
// template <>
// struct hash<cv::KeyPoint> {
//     size_t operator()(const cv::KeyPoint &kp) const {
//         size_t h1 = hash<float>()(kp.pt.x);
//         size_t h2 = hash<float>()(kp.pt.y);
//         size_t h3 = hash<float>()(kp.size);
//         size_t h4 = hash<float>()(kp.angle);
//         size_t h5 = hash<float>()(kp.response);
//         size_t h6 = hash<int>()(kp.octave);
//         size_t h7 = hash<int>()(kp.class_id);

//         return (h1 ^ (h2 << 1)) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
//     }
// };
// }  // namespace std
// namespace cv {
// inline bool operator==(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) {
//     return kp1.pt == kp2.pt && kp1.size == kp2.size && kp1.angle == kp2.angle &&
//            kp1.response == kp2.response && kp1.octave == kp2.octave && kp1.class_id ==
//            kp2.class_id;
// }
// }  // namespace cv
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
    static constexpr char pose_translation_symbol_char = 't';
    static constexpr char pose_bearing_symbol_char = 'b';
    static constexpr char landmark_symbol_char = 'l';
    static constexpr char camera_symbol_char = 'k';

    struct Landmark {
        Landmark(const gtsam::Symbol &lmk_factor_symbol, const gtsam::Symbol &cam_pose_symbol,
                 const gtsam::Point2 &projection, const gtsam::Cal3_S2 K,
                 float initial_depth_guess = 5.0)
            : lmk_factor_symbol(lmk_factor_symbol),
              cam_pose_symbol(cam_pose_symbol),
              projection(projection),
              p_cam_lmk_guess(robot::geometry::deproject(robot::geometry::get_intrinsic_matrix(K),
                                                         projection, initial_depth_guess)){};
        const gtsam::Symbol lmk_factor_symbol;
        const gtsam::Symbol cam_pose_symbol;
        const gtsam::Point2 projection;
        const gtsam::Point3 p_cam_lmk_guess;
    };

    Backend();
    Backend(gtsam::Cal3_S2 K);
    ~Backend(){};

    template <typename T>
    void add_prior_factor(const gtsam::Symbol &symbol, const T &value,
                          const gtsam::SharedNoiseModel &model);

    template <typename T>
    void add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2,
                            const T &value, const gtsam::SharedNoiseModel &model);

    void add_factor_GPS(const gtsam::Symbol &symbol, const gtsam::Point3 &p_world_gps,
                        const gtsam::SharedNoiseModel &model,
                        const gtsam::Rot3 &R_world_cam = gtsam::Rot3::Identity());

    std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> get_obs_for_lmk(
        const gtsam::Symbol &lmk_symbol);
    void add_landmarks(const std::vector<Landmark> &landmarks);
    void add_landmark(const Landmark &landmark);

    void solve_graph();
    typedef int epoch;
    using graph_step_debug_func = std::function<void(const gtsam::Values &, const epoch)>;
    void solve_graph(const int num_steps,
                     std::optional<graph_step_debug_func> inter_debug_func = std::nullopt);

    const gtsam::Values &get_current_initial_values() const { return initial_estimate_; };
    const gtsam::Values &get_result() const { return result_; };
    const gtsam::Cal3_S2 &get_K() const { return *K_; };

    const gtsam::SharedNoiseModel get_lmk_noise() { return landmark_noise_; };
    const gtsam::SharedNoiseModel get_pose_noise() { return pose_noise_; };
    const gtsam::SharedNoiseModel get_translation_noise() { return translation_noise_; };
    const gtsam::SharedNoiseModel get_gps_noise() { return gps_noise_; };

   private:
    gtsam::Cal3_S2::shared_ptr K_;

    gtsam::Values initial_estimate_;
    gtsam::Values result_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::noiseModel::Diagonal::shared_ptr pose_noise_ =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6(0.1, 0.1, 0.1, 0.01, 0.01, 0.01));
    gtsam::noiseModel::Isotropic::shared_ptr translation_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
    gtsam::noiseModel::Isotropic::shared_ptr gps_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(3, 2.);
};

template <>
void Backend::add_prior_factor<gtsam::Pose3>(const gtsam::Symbol &, const gtsam::Pose3 &,
                                             const gtsam::SharedNoiseModel &);

template <>
void Backend::add_prior_factor<gtsam::Point3>(const gtsam::Symbol &, const gtsam::Point3 &,
                                              const gtsam::SharedNoiseModel &);

template <>
void Backend::add_between_factor<gtsam::Pose3>(const gtsam::Symbol &, const gtsam::Symbol &,
                                               const gtsam::Pose3 &,
                                               const gtsam::SharedNoiseModel &);
template <>
void Backend::add_between_factor<gtsam::Rot3>(const gtsam::Symbol &, const gtsam::Symbol &,
                                              const gtsam::Rot3 &, const gtsam::SharedNoiseModel &);

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
    FeatureManager feature_manager_;

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