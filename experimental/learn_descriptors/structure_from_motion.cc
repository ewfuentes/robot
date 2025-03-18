#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "common/geometry/camera.hh"
#include "common/geometry/translate_types.hh"
#include "gtsam/geometry/triangulation.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"

namespace fs = std::filesystem;

namespace geom = robot::geometry;

namespace robot::experimental::learn_descriptors {

const Eigen::Isometry3d StructureFromMotion::T_symlake_boat_cam = []() {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();    
    transform.linear() = (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 1, 0)) *
                                     Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d(0, 0, 1))).toRotationMatrix();
    return transform;
}();

std::string pose_to_string(gtsam::Pose3 pose) {
    std::stringstream ss;
    ss << pose;
    return ss.str();
};

// const gtsam::Pose3 StructureFromMotion::default_initial_pose = []() {
//     Eigen::Isometry3d
// }();
const gtsam::Pose3 StructureFromMotion::default_initial_pose(StructureFromMotion::T_symlake_boat_cam.matrix());
// const gtsam::Pose3 StructureFromMotion::default_initial_pose =
//     gtsam::Pose3(gtsam::Rot3(T_symlake_boat_cam.rotation()), T_symlake_boat_cam.translation());
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
                                         Frontend::MatcherType frontend_matcher) : initial_pose_(initial_pose) {
    frontend_ = Frontend(frontend_extractor, frontend_matcher);
    backend_ = Backend(K);

    K_ = (cv::Mat_<double>(3, 3) << K.fx(), 0, K.px(), 0, K.fy(), K.py(), 0, 0, 1);
    D_ = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    set_initial_pose(initial_pose);
}

void StructureFromMotion::set_initial_pose(gtsam::Pose3 initial_pose) {
    backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, 0), initial_pose, gtsam::noiseModel::Isotropic::Sigma(6, 0));
}

void StructureFromMotion::add_image(const cv::Mat &img, const gtsam::Pose3 &T_world_cam) {
    cv::Mat img_undistorted;
    cv::undistort(img, img_undistorted, K_, D_);
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_and_descriptors =
        frontend_.get_keypoints_and_descriptors(img);
    keypoint_to_landmarks_.push_back(std::unordered_map<cv::KeyPoint, Backend::Landmark>());
    const size_t idx_img_current = get_num_images_added();
    if (idx_img_current > 0) {
        std::vector<cv::DMatch> matches =
            get_matches(img_keypoints_and_descriptors_.back().second,
                        keypoints_and_descriptors.second, Frontend::enforce_bijective_matches);
        // std::vector<cv::DMatch> matches = frontend_.get_matches(
        //     img_keypoints_and_descriptors_.back().second, keypoints_and_descriptors.second);
        // Frontend::enforce_bijective_matches(matches);
        // matches_.push_back(matches);
        // gtsam::Rot3 R_c0_c1(
        //     geom::estimate_c0_c1(
        //         img_keypoints_and_descriptors_.back().first, keypoints_and_descriptors.first,
        //         matches, geom::eigen_mat_to_cv(geom::get_intrinsic_matrix(get_backend().get_K()))).linear());

        const gtsam::Symbol sym_T_w_c0(Backend::pose_symbol_char, idx_img_current - 1);
        const gtsam::Symbol sym_T_w_c1(Backend::pose_symbol_char, idx_img_current);
        // backend_.add_prior_factor(gtsam::Symbol(Backend::pose_symbol_char, get_num_images_added()), gps_prior, backend_.get_translation_noise());
        // gtsam::Pose3 T_w_c0 = backend_.get_current_initial_values().at<gtsam::Pose3>(sym_T_w_c0);
        // gtsam::Pose3 T_c0_c1(
        //     R_c0_c1, 
        //     t_world_gps - T_w_c0.translation());
        // backend_.add_between_factor(
        //     sym_T_w_c0,
        //     sym_T_w_c1, 
        //     T_c0_c1,
        //     backend_.get_pose_noise());        
        backend_.add_factor_GPS(sym_T_w_c1, T_world_cam.translation(), backend_.get_gps_noise(), T_world_cam.rotation());
        // backend_.add_factor_GPS(sym_T_w_c1, T_world_cam.translation(), backend_.get_gps_noise());

        for (const cv::DMatch match : matches) {
            const cv::KeyPoint kpt_cam0 = img_keypoints_and_descriptors_.back().first[match.queryIdx];
            const cv::KeyPoint kpt_cam1 = keypoints_and_descriptors.first[match.trainIdx];

            if (keypoint_to_landmarks_[idx_img_current - 1].find(kpt_cam0) == keypoint_to_landmarks_[idx_img_current - 1].end()) {
                const Backend::Landmark landmark_cam_0(
                    gtsam::Symbol(Backend::landmark_symbol_char, landmark_count_),
                    sym_T_w_c0,
                    gtsam::Point2(
                        static_cast<double>(
                            img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.x),
                        static_cast<double>(
                            img_keypoints_and_descriptors_.back().first[match.queryIdx].pt.y)),
                    backend_.get_K(), 3.0);
                keypoint_to_landmarks_[idx_img_current - 1].emplace(kpt_cam0, landmark_cam_0);
                landmark_count_++;
            } 
            // std::cout << "kpt_cam0 in keypoint_to_landmarks_: " << 
            //     (keypoint_to_landmarks_[idx_img_current - 1].find(kpt_cam0) == keypoint_to_landmarks_[idx_img_current - 1].end()) 
            //     << std::endl;                        
            const Backend::Landmark landmark_cam_0 = keypoint_to_landmarks_[idx_img_current - 1].at(kpt_cam0);

            // technically don't need this conditional when enforcing bijectivity
            if (keypoint_to_landmarks_[idx_img_current].find(kpt_cam1) == keypoint_to_landmarks_[idx_img_current].end()) {
                const Backend::Landmark landmark_cam_1(
                    landmark_cam_0.lmk_factor_symbol,
                    sym_T_w_c1,
                    gtsam::Point2(
                        static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.x),
                        static_cast<double>(keypoints_and_descriptors.first[match.trainIdx].pt.y)),
                    backend_.get_K(), 3.0);
                keypoint_to_landmarks_[idx_img_current].emplace(kpt_cam1, landmark_cam_1);                
            }             
            const Backend::Landmark landmark_cam_1 = keypoint_to_landmarks_[idx_img_current].at(kpt_cam1);

            backend_.add_landmark(landmark_cam_0);
            backend_.add_landmark(landmark_cam_1);
        }
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
    // initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

Backend::Backend(gtsam::Cal3_S2 K) {
    K_ = boost::make_shared<gtsam::Cal3_S2>(K);
    // initial_estimate_.insert(gtsam::Symbol(camera_symbol_char, 0), K);
}

template <>
void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Pose3 &value, const gtsam::SharedNoiseModel &noise) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(symbol, value, noise);
    initial_estimate_.insert_or_assign(symbol, value);
    // std::cout << "adding a prior factor! with symbol: " << symbol << std::endl;
    // initial_estimate_.print("values after adding prior: ");
}

template <>
void Backend::add_prior_factor(const gtsam::Symbol &symbol, const gtsam::Point3 &value, const gtsam::SharedNoiseModel &noise) {
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(symbol, value, noise);
    gtsam::Rot3 R;
    if (initial_estimate_.exists(symbol)) {        
        R = initial_estimate_.at<gtsam::Pose3>(symbol).rotation();                
    }     
    initial_estimate_.insert_or_assign(symbol, gtsam::Pose3(R, value));
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
                                       initial_estimate_.at<gtsam::Pose3>(symbol_1).compose(gtsam::Pose3(value, gtsam::Point3::Zero())));
}

void Backend::add_factor_GPS(const gtsam::Symbol &symbol, const gtsam::Point3 &t_world_cam, const gtsam::SharedNoiseModel &model,
        const gtsam::Rot3 & R_world_cam) {
    
    graph_.emplace_shared<gtsam::GPSFactor>(symbol, t_world_cam, model);
    initial_estimate_.insert_or_assign(symbol, gtsam::Pose3(R_world_cam, t_world_cam));
}

std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> Backend::get_obs_for_lmk(const gtsam::Symbol &lmk_symbol) {
    std::vector<gtsam::Pose3> cam_poses;
    std::vector<gtsam::Point2> observations;

    // Iterate over all factors in the graph
    for (const auto& factor : graph_) {
        auto projFactor = boost::dynamic_pointer_cast<
            gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(factor);
        
        if (projFactor && projFactor->keys().at(1) == lmk_symbol) {
            // Get the camera pose symbol
            gtsam::Symbol cameraSymbol = projFactor->keys().at(0);

            // Retrieve the camera pose from values
            if (!initial_estimate_.exists(cameraSymbol)) continue;
            gtsam::Pose3 cameraPose = initial_estimate_.at<gtsam::Pose3>(cameraSymbol);

            // Get the 2D observation (keypoint measurement)
            gtsam::Point2 observation = projFactor->measured();

            // Store the pose and corresponding observation
            observations.push_back(observation);
            cam_poses.push_back(cameraPose);
        }
    }
    return std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>>(cam_poses, observations);
}


void Backend::add_landmarks(const std::vector<Landmark> &landmarks) {
    for (const Landmark &landmark : landmarks) {
        add_landmark(landmark);
    }
}

void Backend::add_landmark(const Landmark &landmark) {
    graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(landmark.projection, landmark_noise_, landmark.cam_pose_symbol, landmark.lmk_factor_symbol,
        K_);
    if (!initial_estimate_.exists(landmark.cam_pose_symbol)) {
        throw std::runtime_error(
            "landmark.cam_pose_symbol must already exist in Backend.initial_estimate_ before "
            "add_landmark is called.");
    }
    gtsam::Point3 p_world_lmk_estimate =
        initial_estimate_.at<gtsam::Pose3>(landmark.cam_pose_symbol)
            * landmark.p_cam_lmk_guess;
    initial_estimate_.insert_or_assign(landmark.lmk_factor_symbol, p_world_lmk_estimate);

    std::pair<std::vector<gtsam::Pose3>, std::vector<gtsam::Point2>> lmk_obs = get_obs_for_lmk(landmark.lmk_factor_symbol);
    if (lmk_obs.first.size() >= 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            p_world_lmk_estimate = gtsam::triangulatePoint3(
                lmk_obs.first, K_, 
                gtsam::Point2Vector(lmk_obs.second.begin(), lmk_obs.second.end()));
            
            // Optional: perform an explicit cheirality check
            bool valid = true;
            for (const auto &pose : lmk_obs.first) {
                // Transform point to the camera coordinate system.
                // transformTo() converts a world point to the camera frame.
                gtsam::Point3 p_cam = pose.transformTo(p_world_lmk_estimate);
                if (p_cam.z() <= 0) {  // Check that the depth is positive
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                initial_estimate_.update(landmark.lmk_factor_symbol, p_world_lmk_estimate);
            } else {
                std::cerr << "Triangulated landmark failed cheirality check; keeping initial guess." << std::endl;
            }
        } catch (const gtsam::TriangulationCheiralityException &e) {
            // Handle the exception gracefully by logging and retaining the previous estimate.
            std::cerr << "Triangulation Cheirality Exception: " << e.what() 
                      << ". Keeping initial landmark estimate." << std::endl;
        }
    }
}

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}

}  // namespace robot::experimental::learn_descriptors