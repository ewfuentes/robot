#include "experimental/learn_descriptors/structure_from_motion.hh"

namespace robot::experimental::learn_descriptors {
StructureFromMotion::StructureFromMotion(Frontend::ExtractorType frontend_extractor, gtsam::Cal3_S2 K, gtsam::Pose3 initial_pose,
                               Frontend::MatcherType frontend_matcher) {
    frontend_ = Frontend(frontend_extractor, frontend_matcher);
    set_initial_pose(initial_pose);    
    backend_ = Backend(K);
}

void StructureFromMotion::set_initial_pose(gtsam::Pose3 initial_pose) {
    backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, 0), initial_pose);
}

void StructureFromMotion::add_image(const cv::Mat &img) {
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_and_descriptors = frontend_.get_keypoints_and_descriptors(img);   
    landmarks_.push_back(std::vector<Backend::Landmark>());
    if (get_num_images_added() > 0) {
        std::vector<cv::DMatch> matches = frontend_.get_matches(img_keypoints_and_descriptors_.back().second, keypoints_and_descriptors.second);        
        Frontend::enforce_bijective_matches(matches);

        matches_.push_back(matches);
        backend_.add_between_factor(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()-1),
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()),
            gtsam::Pose3::Identity());
        for (const cv::DMatch match : matches) {
            Backend::Landmark landmark_cam_1(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::camera_symbol_char, get_num_images_added()-1),
                gtsam::Point2(
                    static_cast<double>(img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.x),
                    static_cast<double>(img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.y) 
                )
            );
            Backend::Landmark landmark_cam_2(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::camera_symbol_char, get_num_images_added()),
                gtsam::Point2(
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.x),
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.y)
                )
            );
            landmark_count_++;
            landmarks_[get_num_images_added()-1].push_back(landmark_cam_1);
            landmarks_[get_num_images_added()].push_back(landmark_cam_2);
            backend_.add_landmark(landmark_cam_1);
            backend_.add_landmark(landmark_cam_2);
        }        
    }
    img_keypoints_and_descriptors_.push_back(keypoints_and_descriptors);
}

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
    matches.erase(
        std::remove_if(matches.begin(), matches.end(), [dist_threshhold](const cv::DMatch& match) {
            return match.distance > dist_threshhold;
        }),
        matches.end());
}

void Frontend::enforce_bijective_matches(std::vector<cv::DMatch>& matches) {    
    std::unordered_map<int, cv::DMatch> bestQueryMatch;
    std::unordered_map<int, cv::DMatch> bestTrainMatch;
    
    for (const auto& match : matches) {
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
    
    matches.erase(
        std::remove_if(
            matches.begin(), matches.end(),
            [&bestQueryMatch, &bestTrainMatch](const cv::DMatch& match) {
                int queryIdx = match.queryIdx;
                int trainIdx = match.trainIdx;

                return bestQueryMatch[queryIdx].trainIdx != trainIdx ||
                       bestTrainMatch[trainIdx].queryIdx != queryIdx;
            }),
        matches.end());
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

Backend::Backend() {
    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    gtsam::Cal3_S2 K(fx, fy, 0, cx, cy);
    
    initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

Backend::Backend(gtsam::Cal3_S2 K) {
    initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Pose3 &value) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(symbol, value, pose_noise_);
    initial_estimate_.insert(symbol, value);
}

void Backend::add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2, const gtsam::Pose3 &value) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(symbol_1, symbol_2, value, pose_noise_);
    initial_estimate_.insert(symbol_2, initial_estimate_.at<gtsam::Pose3>(symbol_2).compose(value));
}

void Backend::add_landmarks(const std::vector<Landmark> &landmarks) {
    for (const Landmark &landmark : landmarks) {
        add_landmark(landmark);
    }
}

void Backend::add_landmark(const Landmark &landmark) {
    graph_.emplace_shared<gtsam::GeneralSFMFactor2<gtsam::Cal3_S2>>(
        landmark.projection, measurement_noise_, landmark.cam_pose_symbol, landmark.lmk_factor_symbol, gtsam::Symbol(camera_symbol_char, 0) 
    );
    initial_estimate_.insert(landmark.lmk_factor_symbol, landmark.initial_guess);
}

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}
}  // namespace robot::experimental::learn_descriptors