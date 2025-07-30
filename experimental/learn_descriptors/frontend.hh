#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "experimental/learn_descriptors/image_point.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
struct FrontendParams {
    enum class ExtractorType { SIFT, ORB };
    enum class MatcherType { BRUTE_FORCE, KNN, FLANN };
    ExtractorType extractor_type = ExtractorType::SIFT;
    MatcherType matcher_type = MatcherType::KNN;
    bool exhaustive = true;
    bool incremental = false;  // a.k.a. images are successive
};
struct ImageAndPoint {
    cv::Mat image_distorted;
    std::shared_ptr<ImagePoint> shared_img_pt;
};
class Frontend {
   public:
    static constexpr char symbol_pose_char = 'x';
    static constexpr char symbol_lmk_char = 'l';
    explicit Frontend(FrontendParams params);
    ~Frontend(){};

    void populate_frames(bool verbose = false);
    void match_frames_and_build_tracks();
    void interpolate_frames();

    void add_image(const ImageAndPoint &img_and_pt) { images_and_points_.push_back(img_and_pt); };
    void add_images(const std::vector<ImageAndPoint> &img_and_pts) {
        for (const ImageAndPoint &img_and_pt : img_and_pts) {
            images_and_points_.push_back(img_and_pt);
        }
    };

    const FrontendParams::ExtractorType &extractor_type() const { return params_.extractor_type; };
    const FrontendParams::MatcherType &matcher_type() const { return params_.matcher_type; };
    FeatureTracks &feature_tracks() const { return feature_tracks_; };
    const FrameLandmarkIdMap &frame_landmark_id_map() const { return lmk_id_map_; };
    std::vector<SharedFrame> &frames() { return shared_frames_; };

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> extract_features(const cv::Mat &img) const;
    std::vector<cv::DMatch> compute_matches(const cv::Mat &descriptors1,
                                            const cv::Mat &descriptors2) const;
    static void threshold_matches(std::vector<cv::DMatch> &matches, float dist_threshhold);
    static void enforce_bijective_matches(std::vector<cv::DMatch> &matches);
    static void enforce_bijective_buffer_matches(std::vector<cv::DMatch> &matches);
    static void draw_keypoints(const cv::Mat &img, std::vector<cv::KeyPoint> keypoints,
                               cv::Mat &img_keypoints_out) {
        cv::drawKeypoints(img, keypoints, img_keypoints_out, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    static void draw_matches(const cv::Mat &img1, std::vector<cv::KeyPoint> keypoints1,
                             const cv::Mat &img2, std::vector<cv::KeyPoint> keypoints2,
                             std::vector<cv::DMatch> matches, cv::Mat &img_matches_out) {
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches_out);
    }

   private:
    bool compute_brute_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               std::vector<cv::DMatch> &matches_out) const;
    bool compute_KNN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                             std::vector<cv::DMatch> &matches_out) const;
    bool compute_FLANN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               std::vector<cv::DMatch> &matches_out) const;
    FrontendParams params_;

    cv::Ptr<cv::Feature2D> feature_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;

    std::vector<ImageAndPoint> images_and_points_;
    FeatureTracks feature_tracks_;
    FrameLandmarkIdMap lmk_id_map_;
    std::vector<SharedFrame> shared_frames_;
};
}  // namespace robot::experimental::learn_descriptors