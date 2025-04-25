#include "experimental/learn_descriptors/frontend.hh"

#include <unordered_map>

namespace robot::experimental::learn_descriptors {

Frontend::Frontend(ExtractorType frontend_algorithm, MatcherType frontend_matcher) {
    extractor_type_ = frontend_algorithm;
    matcher_type_ = frontend_matcher;

    switch (extractor_type_) {
        case ExtractorType::SIFT:
            feature_extractor_ = cv::SIFT::create();
            break;
        case ExtractorType::ORB:
            feature_extractor_ = cv::ORB::create();
            break;
        default:
            // Error handling needed?
            break;
    }
    switch (matcher_type_) {
        case MatcherType::BRUTE_FORCE:
            descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2);
            break;
        case MatcherType::KNN:
            descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2);
            break;
        case MatcherType::FLANN:
            if (frontend_algorithm == ExtractorType::ORB) {
                throw std::invalid_argument("FLANN can not be used with ORB.");
            }
            descriptor_matcher_ = cv::FlannBasedMatcher::create();
            break;
        default:
            // Error handling needed?
            break;
    }
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> Frontend::get_keypoints_and_descriptors(
    const cv::Mat &img) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    switch (extractor_type_) {
        default:  // the opencv extractors have the same function signature
            feature_extractor_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
            break;
    }
    return std::pair<std::vector<cv::KeyPoint>, cv::Mat>(keypoints, descriptors);
}

std::vector<cv::DMatch> Frontend::get_matches(const cv::Mat &descriptors1,
                                              const cv::Mat &descriptors2) const {
    std::vector<cv::DMatch> matches;
    switch (matcher_type_) {
        case MatcherType::BRUTE_FORCE:
            get_brute_matches(descriptors1, descriptors2, matches);
            break;
        case MatcherType::KNN:
            get_KNN_matches(descriptors1, descriptors2, matches);
            break;
        case MatcherType::FLANN:
            get_FLANN_matches(descriptors1, descriptors2, matches);
            break;
        default:
            break;
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

void Frontend::threshold_matches(std::vector<cv::DMatch> &matches, float dist_threshhold) {
    matches.erase(std::remove_if(matches.begin(), matches.end(),
                                 [dist_threshhold](const cv::DMatch &match) {
                                     return match.distance > dist_threshhold;
                                 }),
                  matches.end());
}

void Frontend::enforce_bijective_matches(std::vector<cv::DMatch> &matches) {
    std::unordered_map<int, cv::DMatch> bestQueryMatch;
    std::unordered_map<int, cv::DMatch> bestTrainMatch;

    for (const auto &match : matches) {
        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;

        if (bestQueryMatch.find(queryIdx) == bestQueryMatch.end() ||
            match.distance < bestQueryMatch[queryIdx].distance) {
            bestQueryMatch[queryIdx] = match;
        }

        if (bestTrainMatch.find(trainIdx) == bestTrainMatch.end() ||
            match.distance < bestTrainMatch[trainIdx].distance) {
            bestTrainMatch[trainIdx] = match;
        }
    }

    matches.erase(std::remove_if(matches.begin(), matches.end(),
                                 [&bestQueryMatch, &bestTrainMatch](const cv::DMatch &match) {
                                     int queryIdx = match.queryIdx;
                                     int trainIdx = match.trainIdx;

                                     return bestQueryMatch[queryIdx].trainIdx != trainIdx ||
                                            bestTrainMatch[trainIdx].queryIdx != queryIdx;
                                 }),
                  matches.end());
}

void Frontend::enforce_bijective_buffer_matches(std::vector<cv::DMatch> &matches) {
    // Store best and second-best matches per query_idx
    std::unordered_map<int, std::pair<cv::DMatch, float>> best_two_query_matches;
    for (const auto &match : matches) {
        int query_idx = match.queryIdx;
        float dist = match.distance;

        if (!best_two_query_matches.count(query_idx)) {
            best_two_query_matches[query_idx] = {match, std::numeric_limits<float>::max()};
        } else {
            auto &[best_match, second_best_dist] = best_two_query_matches[query_idx];
            if (dist < best_match.distance) {
                second_best_dist = best_match.distance;
                best_match = match;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
            }
        }
    }

    // Keep matches where best < 0.5 * second_best
    std::vector<cv::DMatch> filtered;
    for (const auto &[query_idx, pair] : best_two_query_matches) {
        const auto &[best_match, second_best_dist] = pair;
        if (best_match.distance < 0.5f * second_best_dist) {
            filtered.push_back(best_match);
        }
    }

    // Enforce bijection: each train_idx should also be best for only one query_idx
    std::unordered_map<int, cv::DMatch> best_train_match;
    for (const auto &match : filtered) {
        int train_idx = match.trainIdx;
        if (!best_train_match.count(train_idx) ||
            match.distance < best_train_match[train_idx].distance) {
            best_train_match[train_idx] = match;
        }
    }

    matches.clear();
    for (const auto &[train_idx, match] : best_train_match) {
        matches.push_back(match);
    }
}

bool Frontend::get_brute_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                 std::vector<cv::DMatch> &matches_out) const {
    if (matcher_type_ != MatcherType::BRUTE_FORCE) {
        return false;
    }
    matches_out.clear();
    descriptor_matcher_->match(descriptors1, descriptors2, matches_out);
    return true;
}

bool Frontend::get_KNN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               std::vector<cv::DMatch> &matches_out) const {
    if (matcher_type_ != MatcherType::KNN) {
        return false;
    }
    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const float ratio_thresh = 0.7f;
    matches_out.clear();
    // Lowe's Ratio Test
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches_out.push_back(knn_matches[i][0]);
        }
    }
    return true;
}

bool Frontend::get_FLANN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                 std::vector<cv::DMatch> &matches_out) const {
    if (matcher_type_ != MatcherType::FLANN) {
        return false;
    }
    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    const float ratio_thresh = 0.7f;
    matches_out.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches_out.push_back(knn_matches[i][0]);
        }
    }
    return true;
}
}  // namespace robot::experimental::learn_descriptors