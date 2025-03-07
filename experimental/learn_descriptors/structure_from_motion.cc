#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "common/geometry/camera.hh"
#include "common/geometry/translate_types.hh"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"

namespace fs = std::filesystem;

namespace geom = robot::geometry;

namespace robot::experimental::learn_descriptors {

const Eigen::Affine3d StructureFromMotion::T_symlake_boat_cam = []() {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translate(Eigen::Vector3d::Zero());
    transform.rotate(Eigen::Matrix3d(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(1, 0, 0)) *
                                     Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 1, 0))));
    return transform;
}();

std::string pose_to_string(gtsam::Pose3 pose) {
    std::stringstream ss;
    ss << pose;
    return ss.str();
}

// const gtsam::Pose3 StructureFromMotion::default_initial_pose = []() {
//     Eigen::Isometry3d
// }();
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
                                         gtsam::Cal3_S2 K, Eigen::Matrix<double, 5, 1> D,
                                         gtsam::Pose3 initial_pose,
                                         Frontend::MatcherType frontend_matcher) {
    frontend_ = Frontend(frontend_extractor, frontend_matcher);
    backend_ = Backend(K);

    K_ = (cv::Mat_<double>(3, 3) << K.fx(), 0, K.px(), 0, K.fy(), K.py(), 0, 0, 1);
    D_ = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    set_initial_pose(initial_pose);
}

void StructureFromMotion::set_initial_pose(gtsam::Pose3 initial_pose) {
    backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, 0), initial_pose);
}

void StructureFromMotion::add_image(const cv::Mat &img) {
    cv::Mat img_undistorted;
    cv::undistort(img, img_undistorted, K_, D_);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_and_descriptors =
        frontend_.get_keypoints_and_descriptors(img);
    landmarks_.push_back(std::vector<Backend::Landmark>());
    if (get_num_images_added() > 0) {
        std::vector<cv::DMatch> matches =
            get_matches(img_keypoints_and_descriptors_.back().second,
                        keypoints_and_descriptors.second, Frontend::enforce_bijective_matches);
        // std::vector<cv::DMatch> matches = frontend_.get_matches(
        //     img_keypoints_and_descriptors_.back().second, keypoints_and_descriptors.second);
        // Frontend::enforce_bijective_matches(matches);
        // matches_.push_back(matches);
        gtsam::SharedNoiseModel pose_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector6() << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01).finished());
        gtsam::Pose3 between_value(
            geom::estimate_c0_c1(
                img_keypoints_and_descriptors_.back().first, keypoints_and_descriptors.first,
                matches, geom::eigen_mat_to_cv(geom::get_intrinsic_matrix(get_backend().get_K())))
                .matrix());
        backend_.add_between_factor(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1),
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()), between_value,
            pose_noise_model);
        gtsam::Pose3 T_cam_landmark(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 1));
        gtsam::Pose3 T_world_cam1 = backend_.get_current_initial_values().at<gtsam::Pose3>(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1));
        gtsam::Pose3 T_world_cam2 = backend_.get_current_initial_values().at<gtsam::Pose3>(
            gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()));
        for (const cv::DMatch match : matches) {
            Backend::Landmark landmark_cam_1(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added() - 1),
                gtsam::Point2(
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.x),
                    static_cast<double>(
                        img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.y)),
                backend_.get_K(), 3.0);
            Backend::Landmark landmark_cam_2(
                gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()),
                gtsam::Point2(
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.x),
                    static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.y)),
                backend_.get_K(), 3.0);
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

std::vector<cv::DMatch> StructureFromMotion::get_matches(
    const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
    std::optional<StructureFromMotion::MatchFunction> post_process_func) {
    std::vector<cv::DMatch> matches = frontend_.get_matches(descriptors_1, descriptors_2);
    if (post_process_func) {
        (*post_process_func)(matches);
    }
    matches_.push_back(matches);
    return matches;
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
    K_ = boost::make_shared<gtsam::Cal3_S2>(K);
    initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

Backend::Backend(gtsam::Cal3_S2 K) {
    K_ = boost::make_shared<gtsam::Cal3_S2>(K);
    initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Pose3 &value) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(symbol, value, pose_noise_);
    initial_estimate_.insert_or_assign(symbol, value);
    // std::cout << "adding a prior factor! with symbol: " << symbol << std::endl;
    // initial_estimate_.print("values after adding prior: ");
}

template <>
void Backend::add_between_factor<gtsam::Pose3>(const gtsam::Symbol &symbol_1,
                                               const gtsam::Symbol &symbol_2,
                                               const gtsam::Pose3 &value,
                                               const gtsam::SharedNoiseModel &model) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(symbol_1, symbol_2, value, model);
    // std::cout << "adding between factor. symbol_1: " << symbol_1 << ". symbol_2: " << symbol_2 <<
    // std::endl; initial_estimate_.print("values when adding between factor: ");
    initial_estimate_.insert_or_assign(symbol_2,
                                       initial_estimate_.at<gtsam::Pose3>(symbol_1).compose(value));
}

template <>
void Backend::add_between_factor<gtsam::Rot3>(const gtsam::Symbol &symbol_1,
                                              const gtsam::Symbol &symbol_2,
                                              const gtsam::Rot3 &value,
                                              const gtsam::SharedNoiseModel &model) {
    graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(symbol_1, symbol_2, value, model);
    initial_estimate_.insert_or_assign(symbol_2,
                                       initial_estimate_.at<gtsam::Rot3>(symbol_1).compose(value));
}

void Backend::add_landmarks(const std::vector<Landmark> &landmarks) {
    for (const Landmark &landmark : landmarks) {
        add_landmark(landmark);
    }
}

void Backend::add_landmark(const Landmark &landmark) {
    // graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
    //     landmark.projection, landmark_noise_, landmark.cam_pose_symbol,
    //     landmark.lmk_factor_symbol, K_);
    auto factor = boost::make_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
        landmark.projection, landmark_noise_, landmark.cam_pose_symbol, landmark.lmk_factor_symbol,
        K_);
    // Add the factor to the graph.
    graph_.add(factor);
    if (!initial_estimate_.exists(landmark.cam_pose_symbol)) {
        throw std::runtime_error(
            "landmark.cam_pose_symbol must already exist in Backend.initial_estimate_ before "
            "add_landmark is called.");
    }
    gtsam::Point3 p_world_lmk_estimate =
        initial_estimate_.at<gtsam::Pose3>(landmark.cam_pose_symbol)
            .transformTo(landmark.p_cam_lmk_guess);
    if (!initial_estimate_.exists(landmark.lmk_factor_symbol)) {
        initial_estimate_.insert(landmark.lmk_factor_symbol, p_world_lmk_estimate);
    } else {
        initial_estimate_.update(landmark.lmk_factor_symbol, p_world_lmk_estimate);
    }
}

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}

}  // namespace robot::experimental::learn_descriptors