#pragma once

#include <opencv2/opencv.hpp>

namespace robot::experimental::learn_descriptors {
class Frontend {
   public:
    enum class ExtractorType { SIFT, ORB };
    enum class MatcherType { BRUTE_FORCE, KNN, FLANN };

    Frontend(){};
    Frontend(ExtractorType frontend_extractor, MatcherType frontend_matcher);
    ~Frontend(){};

    ExtractorType get_extractor_type() const { return extractor_type_; };
    MatcherType get_matcher_type() const { return matcher_type_; };

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> get_keypoints_and_descriptors(
        const cv::Mat &img) const;
    std::vector<cv::DMatch> get_matches(const cv::Mat &descriptors1,
                                        const cv::Mat &descriptors2) const;

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
class VisualOdometry {
   public:
    VisualOdometry(Frontend::ExtractorType frontend_extractor,
                   Frontend::MatcherType frontend_matcher = Frontend::MatcherType::KNN);
    ~VisualOdometry(){};

   private:
    cv::Mat prev_image_;

    Frontend frontend_;
};
}  // namespace robot::experimental::learn_descriptors