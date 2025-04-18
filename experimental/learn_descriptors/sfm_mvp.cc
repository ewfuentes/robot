#include <utility>

#include "Eigen/Dense"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "experimental/learn_descriptors/symphony_lake_parser.hh"
#include "gtest/gtest.h"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Point2.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "gtsam/geometry/triangulation.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/GPSFactor.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
TEST(SFMMvp, sfm_building_manual) {
    // indices 0-199
    const std::vector<int> indices = []() {
        std::vector<int> tmp;
        for (int i = 0; i < 200; i += 10) {
            tmp.push_back(i);
        }
        return tmp;
    }();
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint img_pt_first = survey.getImagePoint(indices.front());

    // const size_t img_width = img_pt_first.width, img_height = img_pt_first.height;
    const double fx = img_pt_first.fx, fy = img_pt_first.fy;
    const double cx = img_pt_first.cx, cy = img_pt_first.cy;
    gtsam::Cal3_S2::shared_ptr K = boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0, cx, cy);
    Eigen::Matrix<double, 5, 1> D =
        (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1, SymphonyLakeCamParams::k2,
         SymphonyLakeCamParams::p1, SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
            .finished();
    cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K->fx(), 0, K->px(), 0, K->fy(), K->py(), 0, 0, 1);
    cv::Mat D_mat = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    // let world be the first boat base recorded. T_world_camera0 = T_earth_boat0 * T_boat_camera
    Eigen::Isometry3d T_earth_boat0 = DataParser::get_T_world_boat(img_pt_first);
    Eigen::Isometry3d T_world_boat0;
    T_world_boat0.linear() = T_earth_boat0.linear();
    Eigen::Isometry3d T_world_camera0 = T_world_boat0 * DataParser::get_T_boat_camera(img_pt_first);
    // StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
    //                         gtsam::Pose3(T_world_camera0.matrix()));
    Frontend frontend(Frontend::ExtractorType::SIFT, Frontend::MatcherType::KNN);

    gtsam::Values initial_estimate_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::noiseModel::Diagonal::shared_ptr pose_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6(0.1, 0.1, 0.1, 0.01, 0.01, 0.01));
    gtsam::noiseModel::Isotropic::shared_ptr translation_noise =
        gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
    gtsam::noiseModel::Isotropic::shared_ptr gps_noise = gtsam::noiseModel::Isotropic::Sigma(3, 2.);

    std::vector<Frame> frames;
    FrameId id = 0;
    for (const int &idx : indices) {
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        cv::Mat img_undistorted;
        cv::undistort(img, img_undistorted, K_mat, D_mat);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);

        Eigen::Isometry3d T_world_boat = DataParser::get_T_world_boat(img_pt);
        T_world_boat.translation() -= T_earth_boat0.translation();
        Eigen::Isometry3d T_world_cam = T_world_boat * DataParser::get_T_boat_camera(img_pt);
        gtsam::Pose3 T_world_cam_gtsam(T_world_cam.matrix());

        graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', id),
                                                                T_world_cam_gtsam, pose_noise);

        Frame frame(id, img_undistorted, K, gtsam::Pose3(T_world_cam.matrix()));

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> kpts_descs =
            frontend.get_keypoints_and_descriptors(img_undistorted);
        KeypointsCV kpts;
        for (auto kpt : kpts_descs.first) {
            kpts.push_back(kpt.pt);
        }
        frame.add_keypoints(kpts);
        frame.assign_descriptors(kpts_descs.second);

        frames.push_back(frame);

        id++;
    }

    FeatureTracks feature_tracks;
    LandmarkIdMap lmk_id_map;
    LandmarkId lmk_id;
    for (size_t i = 0; i < indices.size() - 1; i++) {
        std::vector<cv::DMatch> matches =
            frontend.get_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
        frontend.enforce_bijective_matches(matches);
        for (const cv::DMatch match : matches) {
            const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
            const KeypointCV kpt_cam1 = frames[i + 1].get_keypoints()[match.trainIdx];

            auto key = std::make_pair(i, kpt_cam0);
            if (lmk_id_map.find(key) != lmk_id_map.end()) {
                auto id = lmk_id_map.at(key);
                feature_tracks.at(id).obs_.emplace_back(i, kpt_cam0);
                feature_tracks.at(id).obs_.emplace_back(i + 1, kpt_cam1);
            } else {
                FeatureTrack feature_track(i, kpt_cam0);
                feature_track.obs_.emplace_back(i + 1, kpt_cam1);
                feature_tracks.emplace(lmk_id, feature_track);
                lmk_id_map.emplace(std::make_pair(i, kpt_cam0), lmk_id);
                lmk_id++;
            }
        }
    }

    std::vector<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>> projection_factors;
    for (const std::pair<LandmarkId, FeatureTrack> &lmk_feat : feature_tracks) {
        // std::vector<FrameId> cam_poses;
        // std::vector<
        LandmarkId lmk_id = lmk_feat.first;
        FeatureTrack feature_track = lmk_feat.second;
        for (const std::pair<FrameId, KeypointCV> &feat_track : feature_track.obs_) {
            graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                feat_track.second, landmark_noise, gtsam::Symbol('x', feat_track.first),
                gtsam::Symbol('l', lmk_id), K);
        }
    }

    if (feature_track.obs_.size() >= 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            gtsam::Point3 p_ = gtsam::triangulatePoint3(
                lmk_obs.first, K,
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
                std::cerr << "Triangulated landmark failed cheirality check; keeping "
                             "initial guess."
                          << std::endl;
            }
        } catch (const gtsam::TriangulationCheiralityException &e) {
            // Handle the exception gracefully by logging and retaining the previous
            // estimate.
            std::cerr << "Triangulation Cheirality Exception: " << e.what()
                      << ". Keeping initial landmark estimate." << std::endl;
        }
    }
}

// const gtsam::Values initial_values = sfm.get_backend().get_current_initial_values();
// sfm.graph_values(initial_values, "initial values");

// std::cout << "Solving for structure!" << std::endl;

// Backend::graph_step_debug_func solve_iter_debug_func = [&sfm](const gtsam::Values &vals,
//                                                               const Backend::epoch iter) {
//     std::cout << "iteration " << iter << " complete!";
//     std::string window_name = "Iteration_" + std::to_string(iter);
//     sfm.graph_values(vals, window_name);
// };
// sfm.solve_structure(5, solve_iter_debug_func);

// std::cout << "Solution complete." << std::endl;

// const gtsam::Values result_values = sfm.get_structure_result();
// sfm.graph_values(result_values, "optimized values");
}
}  // namespace robot::experimental::learn_descriptors