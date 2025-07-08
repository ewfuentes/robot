#include "experimental/learn_descriptors/frontend.hh"

#include <unordered_map>

namespace robot::experimental::learn_descriptors {

Frontend::Frontend(FrontendParams params_) : params_(params_) {
    switch (params_.extractor_type) {
        case FrontendParams::ExtractorType::SIFT:
            feature_extractor_ = cv::SIFT::create();
            break;
        case FrontendParams::ExtractorType::ORB:
            feature_extractor_ = cv::ORB::create();
            break;
        default:
            // Error handling needed?
            break;
    }
    switch (params_.matcher_type) {
        case FrontendParams::MatcherType::BRUTE_FORCE:
            descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2);
            break;
        case FrontendParams::MatcherType::KNN:
            descriptor_matcher_ = cv::BFMatcher::create(cv::NORM_L2);
            break;
        case FrontendParams::MatcherType::FLANN:
            if (params_.extractor_type == FrontendParams::ExtractorType::ORB) {
                throw std::invalid_argument("FLANN can not be used with ORB.");
            }
            descriptor_matcher_ = cv::FlannBasedMatcher::create();
            break;
        default:
            // Error handling needed?
            break;
    }
}

void Frontend::populate_frames() {
    for (const cv::Mat &img : images_) {
    }
}

void Frontend::match_frames_and_build_tracks() {
    if (params_.exhaustive) {
        for (size_t i = 0; i < indices.size() - 1; i += 4) {
            // std::cout << "i: " << i << std::endl;
            for (size_t j = i + 1; j < indices.size(); j++) {
                // std::cout << "j: " << j << std::endl;
                std::vector<cv::DMatch> matches = frontend.compute_matches(
                    frames[i].get_descriptors(), frames[j].get_descriptors());
                // DIAL TO MESS WITH
                frontend.enforce_bijective_buffer_matches(matches);

                if (matches.size() < 5) {
                    continue;
                }

                std::vector<cv::KeyPoint> cv_kpts_1;
                std::vector<cv::KeyPoint> cv_kpts_2;
                for (const KeypointCV &kpt : frames[i].get_keypoints()) {
                    cv::KeyPoint cv_kpt;
                    cv_kpt.pt = kpt;
                    cv_kpts_1.push_back(cv_kpt);
                }
                for (const KeypointCV &kpt : frames[j].get_keypoints()) {
                    cv::KeyPoint cv_kpt;
                    cv_kpt.pt = kpt;
                    cv_kpts_2.push_back(cv_kpt);
                }
                Eigen::Isometry3d world_from_camj(id_to_initial_world_from_cam.at(j).matrix());
                // std::cout << "heartbeat" << std::endl;
                std::optional<Eigen::Isometry3d> scale_cam0_from_cam1 =
                    robot::geometry::estimate_cam0_from_cam1(cv_kpts_1, cv_kpts_2, matches, K_mat);
                if (!scale_cam0_from_cam1) {
                    continue;
                }
                world_from_camj.linear() =
                    (Eigen::Isometry3d(id_to_initial_world_from_cam.at(i).matrix()) *
                     *scale_cam0_from_cam1)
                        .linear();
                // std::cout << "heartbeat 2" << std::endl;
                id_to_initial_world_from_cam.at(j) = gtsam::Pose3(world_from_camj.matrix());
                for (const cv::DMatch match : matches) {
                    const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
                    const KeypointCV kpt_cam1 = frames[j].get_keypoints()[match.trainIdx];

                    auto key = std::make_pair(frames[i].id_, kpt_cam0);
                    if (lmk_id_map.find(key) != lmk_id_map.end()) {
                        auto id = lmk_id_map.at(key);
                        feature_tracks.at(id).obs_.emplace_back(frames[i].id_, kpt_cam0);
                        feature_tracks.at(id).obs_.emplace_back(frames[j].id_, kpt_cam1);
                    } else {
                        FeatureTrack feature_track(i, kpt_cam0);
                        feature_track.obs_.emplace_back(frames[j].id_, kpt_cam1);
                        feature_tracks.emplace(lmk_id, feature_track);
                        lmk_id_map.emplace(std::make_pair(frames[i].id_, kpt_cam0), lmk_id);
                        lmk_id++;
                    }
                }
            }
        }
        std::cout << "done processing matches" << std::endl;
    } else {  // successive only
        for (size_t i = 0; i < indices.size() - 1; i++) {
            std::vector<cv::DMatch> matches = frontend.compute_matches(
                frames[i].get_descriptors(), frames[i + 1].get_descriptors());
            frontend.enforce_bijective_buffer_matches(matches);
            for (const cv::DMatch match : matches) {
                const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
                const KeypointCV kpt_cam1 = frames[i + 1].get_keypoints()[match.trainIdx];

                auto key = std::make_pair(frames[i].id_, kpt_cam0);
                if (lmk_id_map.find(key) != lmk_id_map.end()) {
                    auto id = lmk_id_map.at(key);
                    feature_tracks.at(id).obs_.emplace_back(frames[i].id_, kpt_cam0);
                    feature_tracks.at(id).obs_.emplace_back(frames[i + 1].id_, kpt_cam1);
                } else {
                    FeatureTrack feature_track(i, kpt_cam0);
                    feature_track.obs_.emplace_back(frames[i + 1].id_, kpt_cam1);
                    feature_tracks.emplace(lmk_id, feature_track);
                    lmk_id_map.emplace(std::make_pair(frames[i].id_, kpt_cam0), lmk_id);
                    lmk_id++;
                }
            }
        }
    }
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> Frontend::extract_features(const cv::Mat &img) const {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    switch (params_.extractor_type) {
        default:  // the opencv extractors have the same function signature
            feature_extractor_->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
            break;
    }
    return std::pair<std::vector<cv::KeyPoint>, cv::Mat>(keypoints, descriptors);
}

std::vector<cv::DMatch> Frontend::compute_matches(const cv::Mat &descriptors1,
                                                  const cv::Mat &descriptors2) const {
    std::vector<cv::DMatch> matches;
    switch (params_.matcher_type) {
        case FrontendParams::MatcherType::BRUTE_FORCE:
            compute_brute_matches(descriptors1, descriptors2, matches);
            break;
        case FrontendParams::MatcherType::KNN:
            compute_KNN_matches(descriptors1, descriptors2, matches);
            break;
        case FrontendParams::MatcherType::FLANN:
            compute_FLANN_matches(descriptors1, descriptors2, matches);
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

bool Frontend::compute_brute_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                     std::vector<cv::DMatch> &matches_out) const {
    if (params_.matcher_type != FrontendParams::MatcherType::BRUTE_FORCE) {
        return false;
    }
    matches_out.clear();
    descriptor_matcher_->match(descriptors1, descriptors2, matches_out);
    return true;
}

bool Frontend::compute_KNN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                   std::vector<cv::DMatch> &matches_out) const {
    if (params_.matcher_type != FrontendParams::MatcherType::KNN) {
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

bool Frontend::compute_FLANN_matches(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                     std::vector<cv::DMatch> &matches_out) const {
    if (params_.matcher_type != FrontendParams::MatcherType::FLANN) {
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