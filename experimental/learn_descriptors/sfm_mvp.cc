#include <memory>
#include <utility>

#include "Eigen/Dense"
#include "common/geometry/opencv_viz.hh"
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
class SFMMvpHelper {
 public:
  static bool attempt_triangulate(const std::vector<gtsam::Pose3> &cam_poses,
                                  const std::vector<gtsam::Point2> &cam_obs,
                                  gtsam::Cal3_S2::shared_ptr K,
                                  gtsam::Point3 &out_p_world_landmark) {
    bool valid = true;
    if (cam_poses.size() >= 2) {
      try {
        // Attempt triangulation using DLT (or the GTSAM provided method)
        gtsam::Point3 p_world_lmk = gtsam::triangulatePoint3(
            cam_poses, K, gtsam::Point2Vector(cam_obs.begin(), cam_obs.end()));

        out_p_world_landmark = p_world_lmk;

        // Optional: perform an explicit cheirality check
        for (const auto &pose : cam_poses) {
          // Transform point to the camera coordinate system.
          // transformTo() converts a world point to the camera frame.
          gtsam::Point3 p_cam_lmk = pose.transformTo(p_world_lmk);
          if (p_cam_lmk.z() <= 0) {  // Check that the depth is positive
            valid = false;
            break;
          }
        }
      } catch (const gtsam::TriangulationCheiralityException &e) {
        // Handle the exception gracefully by logging and retaining the previous
        // estimate.
        std::cerr << "Triangulation Cheirality Exception: " << e.what()
                  << ". Keeping initial landmark estimate." << std::endl;
      }
    } else {
      valid = false;
    }
    return valid;
  }

  static void graph_values(
      const gtsam::Values &values, const std::string &window_name,
      const std::vector<gtsam::Symbol> &symbols_pose,
      const std::vector<gtsam::Symbol> &symbols_landmarks) {
    std::vector<Eigen::Isometry3d> final_poses;
    std::vector<Eigen::Vector3d> final_lmks;
    for (const gtsam::Symbol &symbol_pose : symbols_pose) {
      final_poses.emplace_back(values.at<gtsam::Pose3>(symbol_pose).matrix());
    }
    for (const gtsam::Symbol &symbol_lmk : symbols_landmarks) {
      if (!values.exists(symbol_lmk)) {
        std::cout << "WTF " << symbol_lmk << std::endl;
      }
      final_lmks.emplace_back(values.at<gtsam::Point3>(symbol_lmk));
    }
    geometry::viz_scene(final_poses, final_lmks, true, true, window_name);
  }
};

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
  const symphony_lake_dataset::SurveyVector &survey_vector =
      data_parser.get_surveys();
  const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
  const symphony_lake_dataset::ImagePoint img_pt_first =
      survey.getImagePoint(indices.front());

  // const size_t img_width = img_pt_first.width, img_height =
  // img_pt_first.height;
  const double fx = img_pt_first.fx, fy = img_pt_first.fy;
  const double cx = img_pt_first.cx, cy = img_pt_first.cy;
  gtsam::Cal3_S2::shared_ptr K =
      boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0, cx, cy);
  Eigen::Matrix<double, 5, 1> D =
      (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1,
       SymphonyLakeCamParams::k2, SymphonyLakeCamParams::p1,
       SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
          .finished();
  cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K->fx(), 0, K->px(), 0, K->fy(),
                   K->py(), 0, 0, 1);
  cv::Mat D_mat =
      (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

  // let world be the first boat base recorded. T_world_camera0 = T_earth_boat0
  // * T_boat_camera
  Eigen::Isometry3d T_earth_boat0 = DataParser::get_T_world_boat(img_pt_first);
  Eigen::Isometry3d T_world_boat0;
  T_world_boat0.linear() = T_earth_boat0.linear();
  Eigen::Isometry3d T_world_camera0 =
      T_world_boat0 * DataParser::get_T_boat_camera(img_pt_first);
  // StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
  //                         gtsam::Pose3(T_world_camera0.matrix()));
  Frontend frontend(Frontend::ExtractorType::SIFT, Frontend::MatcherType::KNN);

  gtsam::Values initial_estimate_;
  gtsam::NonlinearFactorGraph graph_;

  gtsam::noiseModel::Isotropic::shared_ptr landmark_noise =
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
  gtsam::noiseModel::Diagonal::shared_ptr pose_noise =
      gtsam::noiseModel::Diagonal::Sigmas(
          gtsam::Vector6(0.1, 0.1, 0.1, 0.01, 0.01, 0.01));
  gtsam::noiseModel::Isotropic::shared_ptr translation_noise =
      gtsam::noiseModel::Isotropic::Sigma(2, 0.1);
  gtsam::noiseModel::Isotropic::shared_ptr gps_noise =
      gtsam::noiseModel::Isotropic::Sigma(3, 2.);

  // populate frames and cam_poses
  std::vector<Frame> frames;
  FrameId id = 0;
  std::unordered_map<FrameId, gtsam::Pose3> cam_pose;
  for (const int &idx : indices) {
    const cv::Mat img = survey.loadImageByImageIndex(idx);
    cv::Mat img_undistorted;
    cv::undistort(img, img_undistorted, K_mat, D_mat);
    const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);

    Eigen::Isometry3d T_world_boat = DataParser::get_T_world_boat(img_pt);
    T_world_boat.translation() -= T_earth_boat0.translation();
    Eigen::Isometry3d T_world_cam =
        T_world_boat * DataParser::get_T_boat_camera(img_pt);
    gtsam::Pose3 T_world_cam_gtsam(T_world_cam.matrix());
    cam_pose.emplace(id, T_world_cam_gtsam);

    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', id), T_world_cam_gtsam, pose_noise);

    Frame frame(id, img_undistorted, K, gtsam::Pose3(T_world_cam.matrix()));

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> kpts_descs =
        frontend.get_keypoints_and_descriptors(img_undistorted);
    KeypointsCV kpts;
    for (const cv::KeyPoint &kpt : kpts_descs.first) {
      kpts.push_back(kpt.pt);
    }
    frame.add_keypoints(kpts);
    frame.assign_descriptors(kpts_descs.second);

    frames.push_back(frame);

    id++;
  }

  // populate feature_tracks and lmk_id_map
  FeatureTracks feature_tracks;
  LandmarkIdMap lmk_id_map;
  LandmarkId lmk_id;
  for (size_t i = 0; i < indices.size() - 1; i++) {
    std::vector<cv::DMatch> matches = frontend.get_matches(
        frames[i].get_descriptors(), frames[i + 1].get_descriptors());
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

  // populate graph
  std::vector<gtsam::Symbol> symbols_pose;
  std::vector<gtsam::Symbol> symbols_landmarks;
  for (const std::pair<LandmarkId, FeatureTrack> lmk_feat : feature_tracks) {
    std::vector<gtsam::Pose3> feat_cam_poses;
    std::vector<gtsam::Point2> feat_kpts;
    LandmarkId lmk_id = lmk_feat.first;
    FeatureTrack feature_track = lmk_feat.second;
    const gtsam::Symbol symbol_lmk('l', lmk_id);
    symbols_landmarks.push_back(symbol_lmk);
    for (const std::pair<FrameId, KeypointCV> &feat_track :
         feature_track.obs_) {
      initial_estimate_.insert_or_assign(symbol_lmk, gtsam::Point3(0, 0, 0));
      graph_.emplace_shared<
          gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
          gtsam::Point2(feat_track.second.x, feat_track.second.y),
          landmark_noise, gtsam::Symbol('x', feat_track.first), symbol_lmk, K);

      feat_cam_poses.push_back(cam_pose[feat_track.first]);
      feat_kpts.push_back(
          gtsam::Point2(feat_track.second.x, feat_track.second.y));
    }
    gtsam::Point3 p_wrld_kpt;
    bool triangulate_success = SFMMvpHelper::attempt_triangulate(
        feat_cam_poses, feat_kpts, K, p_wrld_kpt);
    if (triangulate_success) {
      initial_estimate_.insert_or_assign(symbol_lmk, p_wrld_kpt);
    }
  }
  graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gtsam::Symbol('x', 0), cam_pose.at(0), pose_noise);
  initial_estimate_.insert_or_assign(gtsam::Symbol('x', 0), cam_pose.at(0));
  symbols_pose.push_back(gtsam::Symbol('x', 0));
  for (const std::pair<FrameId, gtsam::Pose3> cam_id_pose : cam_pose) {
    if (cam_id_pose.first == 0) {
      continue;
    }
    const gtsam::Symbol key_cam('x', cam_id_pose.first);
    symbols_pose.push_back(key_cam);
    gtsam::Point3 p_wrld_cam = cam_id_pose.second.translation();
    graph_.emplace_shared<gtsam::GPSFactor>(key_cam, p_wrld_cam, gps_noise);
    initial_estimate_.insert_or_assign(key_cam,
                                       gtsam::Pose3(gtsam::Rot3(), p_wrld_cam));
  }

  gtsam::LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
  params.maxIterations = 1;          // We'll manually step it
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_,
                                               params);

  double prev_error = optimizer.error();
  const int num_steps = 5;
  const bool viz_itr = true;
  typedef int epoch;
  std::function<void(const gtsam::Values &, const epoch,
                     const std::vector<gtsam::Symbol> &,
                     const std::vector<gtsam::Symbol> &)>
      graph_itr_debug_func =
          [&](const gtsam::Values &vals, const epoch iter,
              const std::vector<gtsam::Symbol> &symbols_pose,
              const std::vector<gtsam::Symbol> &symbols_landmarks) {
            std::cout << "iteration " << iter << " complete!";
            std::string window_name = "Iteration_" + std::to_string(iter);
            SFMMvpHelper::graph_values(vals, window_name, symbols_pose,
                                       symbols_landmarks);
          };

  SFMMvpHelper::graph_values(initial_estimate_, "Initial Graph", symbols_pose,
                             symbols_landmarks);
  for (int i = 0; i < num_steps; i++) {
    optimizer.iterate();
    double curr_error = optimizer.error();

    if (viz_itr) {
      graph_itr_debug_func(optimizer.values(), i, symbols_pose,
                           symbols_landmarks);
    }
    if (std::abs(prev_error - curr_error) < 1e-6) {
      std::cout << "Converged at iteration " << i << "\n";
      break;
    }
  }
  const gtsam::Values result = optimizer.values();
}

// const gtsam::Values initial_values =
// sfm.get_backend().get_current_initial_values();
// sfm.graph_values(initial_values, "initial values");

// std::cout << "Solving for structure!" << std::endl;

// Backend::graph_step_debug_func solve_iter_debug_func = [&sfm](const
// gtsam::Values &vals,
//                                                               const
//                                                               Backend::epoch
//                                                               iter) {
//     std::cout << "iteration " << iter << " complete!";
//     std::string window_name = "Iteration_" + std::to_string(iter);
//     sfm.graph_values(vals, window_name);
// };
// sfm.solve_structure(5, solve_iter_debug_func);

// std::cout << "Solution complete." << std::endl;

// const gtsam::Values result_values = sfm.get_structure_result();
// sfm.graph_values(result_values, "optimized values");
}  // namespace robot::experimental::learn_descriptors