#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace robot::experimental::learn_descriptors {

const Eigen::Affine3d StructureFromMotion::T_symlake_boat_cam = []() {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translate(Eigen::Vector3d::Zero());
    transform.rotate(Eigen::Matrix3d(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(1, 0, 0)) *
                                     Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 1, 0))));

    // std::stringstream ss;
    // ss << gtsam::Pose3(gtsam::Rot3(transform.rotation()),
    // gtsam::Point3(transform.translation())); json json_obj; json_obj["T_symlake_boat_cam"] =
    // ss.str(); fs::path output_dir = "output"; fs::path output_file = output_dir /
    // "output_sfm.json"; if (!fs::exists(output_dir)) {
    //     fs::create_directory(output_dir);
    // }
    // std::ofstream file(output_file);
    // if (file.is_open()) {
    //     file << json_obj.dump(4);
    //     file.close();
    // }

    return transform;
}();

std::string pose_to_string(gtsam::Pose3 pose) {
    std::stringstream ss;
    ss << pose;
    return ss.str();
}

const gtsam::Pose3 StructureFromMotion::default_initial_pose =
    gtsam::Pose3(gtsam::Rot3(T_symlake_boat_cam.rotation()), T_symlake_boat_cam.translation());
// gtsam::Pose3(
//     gtsam::Rot3(Eigen::Matrix3d(
//         Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix() *
//         Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 1, 0)).toRotationMatrix()
//     )),
//     gtsam::Point3::Identity()
// );

StructureFromMotion::StructureFromMotion(Frontend::ExtractorType frontend_extractor,
                                         gtsam::Cal3_S2 K, gtsam::Pose3 initial_pose,
                                         Frontend::MatcherType frontend_matcher) {
    frontend_ = Frontend(frontend_extractor, frontend_matcher);
    backend_ = Backend(K);

    set_initial_pose(initial_pose);
    // backend_.get_current_initial_values().print("Ooga: ");
}

void StructureFromMotion::set_initial_pose(gtsam::Pose3 initial_pose) {
    backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, 0), initial_pose);
}

void StructureFromMotion::add_image(const cv::Mat &img) {
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_and_descriptors =
        frontend_.get_keypoints_and_descriptors(img);
    landmarks_.push_back(std::vector<Backend::Landmark>());
    if (get_num_images_added() > 0) {
        std::vector<cv::DMatch> matches = frontend_.get_matches(
            img_keypoints_and_descriptors_.back().second, keypoints_and_descriptors.second);
        Frontend::enforce_bijective_matches(matches);

        matches_.push_back(matches);
        gtsam::Pose3 between_value = get_backend().estimate_c0_c1(
            img_keypoints_and_descriptors_.back().first, keypoints_and_descriptors.first, matches,
            get_backend().get_K());
        backend_.add_between_factor(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1),
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()), between_value);
        gtsam::Pose3 T_cam_landmark(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 1));
        gtsam::Pose3 T_world_cam1 = backend_.get_current_initial_values().at<gtsam::Pose3>(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1));
        gtsam::Pose3 T_world_cam2 = backend_.get_current_initial_values().at<gtsam::Pose3>(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()));
        int count = 0;
        for (const cv::DMatch match : matches) {
            if (count < 1) {
                // std::cout << "hello???" << std::endl;
                json json_obj;
                json_obj["add_image_1"] = {
                    {"T_world_cam1", pose_to_string(T_world_cam1)},
                    {"T_world_cam2", pose_to_string(T_world_cam1)},
                    {"T_world_lmkcam1", pose_to_string(T_world_cam1 * T_cam_landmark)},
                    {"T_world_lmkcam2", pose_to_string(T_world_cam2 * T_cam_landmark)}};
                fs::path output_dir = "output";
                fs::path output_file = output_dir / "output_sfm.json";
                if (!fs::exists(output_dir)) {
                    fs::create_directory(output_dir);
                }
                std::ofstream file(output_file);
                if (file.is_open()) {
                    file << json_obj.dump(4);
                    file.close();
                }
            }
            count++;
            // std::cout << "T_world_landmark" <<  T_world_cam1 * T_cam_landmark << std::endl;
            Backend::Landmark landmark_cam_1(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1),
                gtsam::Point2(
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.x),
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.y)),
                (T_world_cam1 * T_cam_landmark).translation());
            Backend::Landmark landmark_cam_2(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()),
                gtsam::Point2(
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.x),
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.y)),
                (T_world_cam2 * T_cam_landmark).translation());
            landmarks_[get_num_images_added() - 1].push_back(landmark_cam_1);
            landmarks_[get_num_images_added()].push_back(landmark_cam_2);
            backend_.add_landmark(landmark_cam_1);
            backend_.add_landmark(landmark_cam_2);
            landmark_count_++;
        }
        // std::cout << "number of landmarks: " << count << std::endl;
        // std::cout << "number of landmarks added to graph: " << [this](){int count = 0; for (const
        // auto &landmark : landmarks_){count += landmark.size();} return count; }() << std::endl;
        // backend_.get_current_initial_values().print("Current initial values: ");
        // std::cout << "landmark_count_: " << landmark_count_ << std::endl;
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
    // std::cout << "adding a prior factor! with symbol: " << symbol << std::endl;
    // initial_estimate_.print("values after adding prior: ");
}

template <>
void Backend::add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2,
                                 const gtsam::Pose3 &value, const gtsam::SharedNoiseModel model) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(symbol_1, symbol_2, value,
                                                              model);
    // std::cout << "adding between factor. symbol_1: " << symbol_1 << ". symbol_2: " << symbol_2 <<
    // std::endl; initial_estimate_.print("values when adding between factor: ");
    initial_estimate_.insert(symbol_2, initial_estimate_.at<gtsam::Pose3>(symbol_1).compose(value));
}

template <>
void Backend::add_between_factor(const gtsam::Symbol &symbol_1, const gtsam::Symbol &symbol_2,
                                 const gtsam::Rot3 &value, const gtsam::SharedNoiseModel &model) {
    graph_.empalce_shared<gtsam::BetweenFactor<gtsam::Rot3>>(symbol_1, symbol_2, value, model);
    initial_estimate_.insert()
}

void Backend::add_landmarks(const std::vector<Landmark> &landmarks) {
    for (const Landmark &landmark : landmarks) {
        add_landmark(landmark);
    }
}

void Backend::add_landmark(const Landmark &landmark) {
    graph_.emplace_shared<gtsam::GeneralSFMFactor2<gtsam::Cal3_S2>>(
        landmark.projection, landmark_noise_, landmark.cam_pose_symbol,
        landmark.lmk_factor_symbol, gtsam::Symbol(camera_symbol_char, 0));
    if (!initial_estimate_.exists(landmark.lmk_factor_symbol)) {
        initial_estimate_.insert(landmark.lmk_factor_symbol, landmark.initial_guess);
    } else {
        initial_estimate_.update(landmark.lmk_factor_symbol, landmark.initial_guess);
    }
}

gtsam::Pose3 Backend::estimate_c0_c1(const std::vector<cv::KeyPoint> &kpts1,
                                    const std::vector<cv::KeyPoint> &kpts2,
                                    const std::vector<cv::DMatch> &matches,
                                    const gtsam::Cal3_S2 &K) {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (const cv::DMatch &match : matches) {
        std::cout << "query point: " << kpts1[match.queryIdx].pt << std::endl;
        std::cout << "trian point: " << kpts2[match.trainIdx].pt << std::endl;
        pts1.push_back(kpts1[match.queryIdx].pt);
        pts2.push_back(kpts2[match.trainIdx].pt);
    }
    cv::Mat cv_K = (cv::Mat_<double>(3, 3) << K.fx(), K.skew(), K.px(), 0, K.fy(), K.py(), 0, 0, 1);
    cv::Mat E = cv::findEssentialMat(pts1, pts2, cv_K, cv::RANSAC, 0.999, 1.0);
    cv::Mat R_c1_c0, t_c1_c0;
    cv::recoverPose(E, pts1, pts2, cv_K, R_c1_c0, t_c1_c0);
    gtsam::Point3 t_c1_c0_eigen(t_c1_c0.at<double>(0, 0), t_c1_c0.at<double>(1, 0), t_c1_c0.at<double>(2, 0));
    gtsam::Matrix3 R_c1_c0_eigen;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_c1_c0_eigen(i, j) = R_c1_c0.at<double>(i, j);
        }
    }
    return gtsam::Pose3(gtsam::Rot3(R_c1_c0_eigen), t_c1_c0_eigen).inverse();
}

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}

// json SFM_Logger::gtsam_pose3_to_json(const gtsam::Pose3 &pose) {
//     json json_obj;
//     json_obj = {
//         {},

//     };

//     return json_obj;
// }

// void SFM_Logger::values_to_json(const gtsam::Values &values, const std::filesystem::path
// &output_path) {
//     json json_obj;
//     json_obj["add_image_1"] = {
//         {"T_world_cam1", pose_to_string(T_world_cam1)},
//         {"T_world_cam2", pose_to_string(T_world_cam1)},
//         {"T_world_lmkcam1", pose_to_string(T_world_cam1 * T_cam_landmark)},
//         {"T_world_lmkcam2", pose_to_string(T_world_cam2 * T_cam_landmark)}
//     };
//     if (!fs::exists(output_path)) {
//         fs::create_directory(output_path);
//     }
//     std::ofstream file(output_file);
//     if (file.is_open()) {
//         file << json_obj.dump(4);
//         file.close();
//     }
// }

}  // namespace robot::experimental::learn_descriptors