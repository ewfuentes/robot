#include "experimental/learn_descriptors/frontend.hh"

#include <cmath>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/check.hh"
#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/image_point.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/triangulation.h"
#include "opencv2/opencv.hpp"

namespace detail_sfm {

std::optional<gtsam::Point3> attempt_triangulate(const std::vector<gtsam::Pose3> &cam_poses,
                                                 const std::vector<gtsam::Point2> &cam_obs,
                                                 gtsam::Cal3_S2::shared_ptr K,
                                                 const double max_reproj_error = 2.0,
                                                 const bool verbose = true) {
    gtsam::Point3 p_lmk_in_world;
    if (cam_poses.size() >= 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            p_lmk_in_world = gtsam::triangulatePoint3(
                cam_poses, K, gtsam::Point2Vector(cam_obs.begin(), cam_obs.end()));

        } catch (const gtsam::TriangulationCheiralityException &e) {
            // Handle the exception gracefully by logging and retaining the previous
            // estimate.
            if (verbose)
                std::cerr << "attempt_triangulation failed. Likely cheirality exception: "
                          << e.what() << ". Discarding involved keypoints." << std::endl;
        }
    }
    // Optional: perform an explicit cheirality check
    for (const auto &pose : cam_poses) {
        // Transform point to the camera coordinate system.
        // transformTo() converts a world point to the camera frame.
        gtsam::Point3 p_cam_lmk = pose.transformTo(p_lmk_in_world);
        if (p_cam_lmk.z() <= 0) {  // Check that the depth is positive
            return std::nullopt;
        }
    }

    // Cheirality & reprojection checks
    for (size_t i = 0; i < cam_poses.size(); ++i) {
        const auto &pose = cam_poses[i];
        // Cheirality
        gtsam::Point3 p_cam = pose.transformTo(p_lmk_in_world);
        if (p_cam.z() <= 0) {
            if (verbose) {
                std::cerr << "[triangulate] point behind camera " << i << " (z=" << p_cam.z()
                          << ")\n";
            }
            return std::nullopt;
        }
        // Reprojection error
        if (max_reproj_error > 0) {
            gtsam::PinholeCamera<gtsam::Cal3_S2> cam(pose, *K);
            const auto reproj = cam.project(p_lmk_in_world);
            const double err = (reproj - cam_obs[i]).norm();
            if (err > max_reproj_error) {
                if (verbose) {
                    std::cerr << "[triangulate] reprojection error too large on view " << i << " ("
                              << err << " px)\n";
                }
                return std::nullopt;
            }
        }
    }

    return p_lmk_in_world;
}

gtsam::Pose3 averagePoses(const std::vector<gtsam::Pose3> &poses, int maxIter = 10) {
    if (poses.empty()) throw std::runtime_error("Empty pose vector");

    gtsam::Pose3 mean = poses[0];

    for (int iter = 0; iter < maxIter; ++iter) {
        gtsam::Vector6 total = gtsam::Vector6::Zero();

        for (const auto &pose : poses) {
            gtsam::Pose3 delta = mean.between(pose);
            total += gtsam::Pose3::Logmap(delta);
        }

        total /= poses.size();
        mean = mean.compose(gtsam::Pose3::Expmap(total));
    }

    return mean;
}

gtsam::Point3 averagePoints(const std::vector<gtsam::Point3> &points) {
    if (points.empty()) throw std::runtime_error("Empty point vector");
    gtsam::Point3 sum(0, 0, 0);
    for (const auto &pt : points) sum += pt;
    return sum / points.size();
}

gtsam::Point3 get_variance(const std::vector<gtsam::Point3> &points) {
    const gtsam::Point3 mean = averagePoints(points);
    gtsam::Point3 var(0, 0, 0);
    for (const gtsam::Point3 &pt : points) {
        var += (mean - pt).array().square().matrix();
    }
    return var / points.size();
}

double rotation_error(const Eigen::Isometry3d &T_est, const Eigen::Isometry3d &T_gt) {
    Eigen::Matrix3d R_err = T_gt.rotation().transpose() * T_est.rotation();

    // 2. Compute trace and clamp to [-1,1] for numerical safety
    double tr = R_err.trace();
    double cos_theta = std::min(1.0, std::max(-1.0, (tr - 1.0) / 2.0));

    // 3. Recover angle (in radians)
    return std::acos(cos_theta);
}

// const std::vector<double> get_absolute_trajectory_error(const std::vector<Eigen::Vector3d> &)
// {}
}  // namespace detail_sfm

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
    FrameId id = frames_.size();
    for (const ImageAndPoint &img_and_pt : images_and_points_) {
        const ImagePoint &img_pt = img_and_pt.img_pt;
        cv::Mat img_undistorted;
        cv::fisheye::undistortImage(img_and_pt.image_distorted, img_undistorted, img_pt.K->k_mat(),
                                    img_pt.K->d_mat(), img_pt.K->k_mat(),
                                    img_and_pt.image_distorted.size());
        auto [kpts, descriptors] = extract_features(img_undistorted);
        KeypointsCV kpts_cv;
        for (const cv::KeyPoint &kpt : kpts) {
            kpts_cv.push_back(kpt.pt);
        }

        gtsam::Cal3_S2::shared_ptr K = boost::make_shared<gtsam::Cal3_S2>(
            img_pt.K->fx, img_pt.K->fy, 0, img_pt.K->cx, img_pt.K->cy);

        if (!K) std::cout << "UH OH" << std::endl;

        Frame frame(id, img_undistorted, K, kpts_cv, descriptors);
        const std::optional<Eigen::Isometry3d> maybe_world_from_cam_grnd_trth =
            img_pt.world_from_cam_ground_truth();
        if (maybe_world_from_cam_grnd_trth) {
            frame.world_from_cam_groundtruth_ =
                gtsam::Pose3(maybe_world_from_cam_grnd_trth->matrix());
        }
        const std::optional<Eigen::Vector3d> maybe_cam_in_world = img_pt.cam_in_world();
        if (maybe_cam_in_world) {
            frame.cam_in_world_initial_guess_ = *maybe_cam_in_world;
        }
        if (id == 0) frame.world_from_cam_initial_guess_ = gtsam::Rot3::Identity();
        const std::optional<gtsam::Matrix3> maybe_translation_covariance_in_cam =
            img_pt.translation_covariance_in_cam();
        if (maybe_translation_covariance_in_cam) {
            frame.translation_covariance_in_cam_ = *maybe_translation_covariance_in_cam;
        }
        frames_.push_back(frame);
        id++;
    }
}

void Frontend::match_frames_and_build_tracks() {
    if (params_.exhaustive) {
        for (size_t i = 0; i < frames_.size() - 1; i++) {
            for (size_t j = i + 1; j < frames_.size(); j++) {
                std::vector<cv::DMatch> matches =
                    compute_matches(frames_[i].descriptors(), frames_[j].descriptors());
                enforce_bijective_buffer_matches(matches);
                if (matches.size() < 5) {
                    continue;
                }

                std::vector<cv::KeyPoint> cv_kpts_1;
                std::vector<cv::KeyPoint> cv_kpts_2;
                for (const KeypointCV &kpt : frames_[i].keypoint()) {
                    cv::KeyPoint cv_kpt;
                    cv_kpt.pt = kpt;
                    cv_kpts_1.push_back(cv_kpt);
                }
                for (const KeypointCV &kpt : frames_[j].keypoint()) {
                    cv::KeyPoint cv_kpt;
                    cv_kpt.pt = kpt;
                    cv_kpts_2.push_back(cv_kpt);
                }
                std::optional<Eigen::Isometry3d> scale_cam0_from_cam1 =
                    robot::geometry::estimate_cam0_from_cam1(
                        cv_kpts_1, cv_kpts_2, matches,
                        images_and_points_[i]
                            .img_pt.K->k_mat());  // fine for now since all cameras have the same K,
                                                  // how to reconcile later though...?

                if (!scale_cam0_from_cam1) {
                    continue;
                }
                ROBOT_CHECK(frames_[i].world_from_cam_initial_guess_,
                            "This rotation should be populated.");
                // this could use some work to verify the quality of the output, particularly inside
                // of estiamte_cam0_from_cam1
                // also, at the moment I am not accounting for any covariance between the gps
                // measurement and the unit translation vector from estimate_cam0_from_cam1

                // can try averaging poses here as well
                frames_[j].world_from_cam_initial_guess_.emplace(
                    frames_[i].world_from_cam_initial_guess_->matrix() *
                    scale_cam0_from_cam1->linear().matrix());
                for (const cv::DMatch match : matches) {
                    const KeypointCV kpt_cam0 = frames_[i].keypoint()[match.queryIdx];
                    const KeypointCV kpt_cam1 = frames_[j].keypoint()[match.trainIdx];

                    auto key = std::make_pair(frames_[i].id_, kpt_cam0);
                    if (lmk_id_map_.find(key) != lmk_id_map_.end()) {
                        const size_t id = lmk_id_map_.at(key);
                        ROBOT_CHECK(id < feature_tracks_.size(),
                                    "lmk_id_map_ id's shuold not exceed feature_tracks_ size!");
                        feature_tracks_[id].obs_.emplace_back(frames_[i].id_, kpt_cam0);
                        feature_tracks_[id].obs_.emplace_back(frames_[j].id_, kpt_cam1);
                    } else {
                        FeatureTrack feature_track(i, kpt_cam0);
                        feature_track.obs_.emplace_back(frames_[j].id_, kpt_cam1);
                        feature_tracks_.push_back(feature_track);
                        lmk_id_map_.emplace(std::make_pair(frames_[i].id_, kpt_cam0),
                                            feature_tracks_.size() - 1);
                    }
                }
            }
        }
    } else {  // successive only
        for (size_t i = 0; i < frames_.size() - 1; i++) {
            std::vector<cv::DMatch> matches =
                compute_matches(frames_[i].descriptors(), frames_[i + 1].descriptors());
            enforce_bijective_buffer_matches(matches);
            if (matches.size() < 5) {
                continue;
            }

            std::vector<cv::KeyPoint> cv_kpts_1;
            std::vector<cv::KeyPoint> cv_kpts_2;
            for (const KeypointCV &kpt : frames_[i].keypoint()) {
                cv::KeyPoint cv_kpt;
                cv_kpt.pt = kpt;
                cv_kpts_1.push_back(cv_kpt);
            }
            for (const KeypointCV &kpt : frames_[i + 1].keypoint()) {
                cv::KeyPoint cv_kpt;
                cv_kpt.pt = kpt;
                cv_kpts_2.push_back(cv_kpt);
            }
            std::optional<Eigen::Isometry3d> scale_cam0_from_cam1 =
                robot::geometry::estimate_cam0_from_cam1(cv_kpts_1, cv_kpts_2, matches,
                                                         images_and_points_[i].img_pt.K->k_mat());
            if (!scale_cam0_from_cam1) {
                continue;
            }
            ROBOT_CHECK(frames_[i].world_from_cam_initial_guess_,
                        "This rotation should be populated.");
            frames_[i + 1].world_from_cam_initial_guess_.emplace(
                frames_[i].world_from_cam_initial_guess_->matrix() *
                scale_cam0_from_cam1->linear().matrix());

            for (const cv::DMatch match : matches) {
                const KeypointCV kpt_cam0 = frames_[i].keypoint()[match.queryIdx];
                const KeypointCV kpt_cam1 = frames_[i + 1].keypoint()[match.trainIdx];

                auto key = std::make_pair(frames_[i].id_, kpt_cam0);
                if (lmk_id_map_.find(key) != lmk_id_map_.end()) {
                    const size_t id = lmk_id_map_.at(key);
                    ROBOT_CHECK(id < feature_tracks_.size(),
                                "lmk_id_map_ id's shuold not exceed feature_tracks_ size!");
                    feature_tracks_[id].obs_.emplace_back(frames_[i].id_, kpt_cam0);
                    feature_tracks_[id].obs_.emplace_back(frames_[i + 1].id_, kpt_cam1);
                } else {
                    FeatureTrack feature_track(i, kpt_cam0);
                    feature_track.obs_.emplace_back(frames_[i + 1].id_, kpt_cam1);
                    feature_tracks_.push_back(feature_track);
                    lmk_id_map_.emplace(std::make_pair(frames_[i].id_, kpt_cam0),
                                        feature_tracks_.size() - 1);
                }
            }
        }
    }
    std::cout << "done processing matches" << std::endl;
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