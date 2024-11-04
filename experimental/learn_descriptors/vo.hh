#pragma once

#include <opencv2/opencv.hpp>

namespace robot::experimental::learn_descriptors::vo {
class Frontend {
   public:
    enum class ExtractorType { SIFT, ORB };
    enum class MatcherType { BRUTE_FORCE, KNN, FLANN };

    Frontend(){};
    Frontend(ExtractorType frontend_extractor, MatcherType frontend_matcher);
    ~Frontend(){};

    ExtractorType getExtractorType() const { return _extractor_type; };
    MatcherType getMatcherType() const { return _matcher_type; };

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> getKeypointsAndDescriptors(
        const cv::Mat &img) const;
    std::vector<cv::DMatch> getMatches(const cv::Mat &descriptors1,
                                       const cv::Mat &descriptors2) const;

    static void drawKeypoints(const cv::Mat &img, std::vector<cv::KeyPoint> keypoints,
                              cv::Mat img_keypoints_out) {
        cv::drawKeypoints(img, keypoints, img_keypoints_out, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
    static void drawMatches(const cv::Mat &img1, std::vector<cv::KeyPoint> keypoints1,
                            const cv::Mat &img2, std::vector<cv::KeyPoint> keypoints2,
                            std::vector<cv::DMatch> matches, cv::Mat img_matches_out) {
        cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches_out);
    }

   private:
    bool getBruteMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                         std::vector<cv::DMatch> &matches_out) const;
    bool getKNNMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                       std::vector<cv::DMatch> &matches_out) const;
    bool getFLANNMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                         std::vector<cv::DMatch> &matches_out) const;
    ExtractorType _extractor_type;
    MatcherType _matcher_type;

    cv::Ptr<cv::Feature2D> _feature_extractor;
    cv::Ptr<cv::DescriptorMatcher> _descriptor_matcher;
};
class VO {
   public:
    VO(Frontend::ExtractorType frontend_extractor,
       Frontend::MatcherType frontend_matcher = Frontend::MatcherType::KNN);
    ~VO(){};

   private:
    cv::Mat prev_image;

    Frontend _frontend;
};
}  // namespace robot::experimental::learn_descriptors::vo