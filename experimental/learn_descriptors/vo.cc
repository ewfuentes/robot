#include "experimental/learn_descriptors/vo.hh"

namespace robot::experimental::learn_descriptors::vo {
VO::VO(Frontend::ExtractorType frontend_extractor, Frontend::MatcherType frontend_matcher) {
    _frontend = Frontend(frontend_extractor, frontend_matcher);
}

Frontend::Frontend(ExtractorType frontend_algorithm, MatcherType frontend_matcher) {
    _extractor_type = frontend_algorithm;
    _matcher_type = frontend_matcher;

    switch (_extractor_type) {
        case ExtractorType::SIFT:
            _feature_extractor = cv::SIFT::create();
            break;
        case ExtractorType::ORB:
            _feature_extractor = cv::ORB::create();
            break;
        default:
            // Error handling needed?
            break;
    }
    switch (_matcher_type) {
        case MatcherType::BRUTE_FORCE:
            _descriptor_matcher = cv::BFMatcher::create(cv::NORM_L2);
        case MatcherType::KNN:
            _descriptor_matcher = cv::BFMatcher::create(cv::NORM_L2);
            break;
        case MatcherType::FLANN:
            if (frontend_algorithm == ExtractorType::ORB) {
                throw std::invalid_argument("FLANN can not be used with ORB.");
            }
            _descriptor_matcher = cv::FlannBasedMatcher::create();
        default:
            // Error handling needed?
            break;
    }
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> Frontend::getKeypointsAndDescriptors(
    const cv::Mat &img) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    switch (_extractor_type) {
        default:  // the opencv extractors have the same function signature
            _feature_extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
            break;
    }
    return std::pair<std::vector<cv::KeyPoint>, cv::Mat>(keypoints, descriptors);
}

std::vector<cv::DMatch> Frontend::getMatches(const cv::Mat &descriptors1,
                                             const cv::Mat &descriptors2) const {
    std::vector<cv::DMatch> matches;
    switch (_matcher_type) {
        case MatcherType::BRUTE_FORCE:
            getBruteMatches(descriptors1, descriptors2, matches);
            break;
        case MatcherType::KNN:
            getKNNMatches(descriptors1, descriptors2, matches);
            break;
        case MatcherType::FLANN:
            getFLANNMatches(descriptors1, descriptors2, matches);
        default:
            break;
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

bool Frontend::getBruteMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               std::vector<cv::DMatch> &matches_out) const {
    if (_matcher_type != MatcherType::BRUTE_FORCE) {
        return false;
    }
    matches_out.clear();
    _descriptor_matcher->match(descriptors1, descriptors2, matches_out);
    return true;
}

bool Frontend::getKNNMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                             std::vector<cv::DMatch> &matches_out) const {
    if (_matcher_type != MatcherType::KNN) {
        return false;
    }
    std::vector<std::vector<cv::DMatch>> knn_matches;
    _descriptor_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const float ratio_thresh = 0.7f;
    matches_out.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches_out.push_back(knn_matches[i][0]);
        }
    }
    return true;
}

bool Frontend::getFLANNMatches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               std::vector<cv::DMatch> &matches_out) const {
    if (_matcher_type != MatcherType::FLANN) {
        return false;
    }
    std::vector<std::vector<cv::DMatch>> knn_matches;
    _descriptor_matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const float ratio_thresh = 0.7f;
    matches_out.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches_out.push_back(knn_matches[i][0]);
        }
    }
    return true;
}
}  // namespace robot::experimental::learn_descriptors::vo