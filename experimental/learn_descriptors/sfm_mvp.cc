#include <memory>
#include <utility>

#include "Eigen/Dense"
#include "boost/make_shared.hpp"
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
#include "visualization/opencv/opencv_viz.hh"

namespace std {
template <>
struct hash<gtsam::Symbol> {
    size_t operator()(const gtsam::Symbol &s) const { return hash<uint64_t>()(s.key()); }
};
}  // namespace std

namespace robot::experimental::learn_descriptors {
class SFMMvpHelper {
   public:
    // struct SymbolHasher {
    //     size_t operator()(const gtsam::Symbol &s) const { return std::hash<uint64_t>()(s.key());
    //     }
    // };

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

    static void graph_values(const gtsam::Values &values, const std::string &window_name,
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
        geometry::viz_scene(final_poses, final_lmks, cv::viz::Color::black(), true, true,
                            window_name);
    }

    static gtsam::Values optimize_graph(const gtsam::NonlinearFactorGraph &graph,
                                        const gtsam::Values &values,
                                        const std::vector<gtsam::Symbol> &symbols_pose,
                                        const std::vector<gtsam::Symbol> &symbols_landmarks,
                                        bool viz_itr = false) {
        gtsam::LevenbergMarquardtParams params;
        params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
        params.maxIterations = 1;
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);

        double prev_error = optimizer.error();
        const int num_steps = 5;
        typedef int epoch;
        std::function<void(const gtsam::Values &, const epoch, const std::vector<gtsam::Symbol> &,
                           const std::vector<gtsam::Symbol> &)>
            graph_itr_debug_func = [&](const gtsam::Values &vals, const epoch iter,
                                       const std::vector<gtsam::Symbol> &symbols_pose,
                                       const std::vector<gtsam::Symbol> &symbols_landmarks) {
                std::cout << "iteration " << iter << " complete!";
                std::string window_name = "Iteration_" + std::to_string(iter);
                SFMMvpHelper::graph_values(vals, window_name, symbols_pose, symbols_landmarks);
            };

        for (int i = 0; i < num_steps; i++) {
            optimizer.iterate();
            double curr_error = optimizer.error();

            if (viz_itr) {
                graph_itr_debug_func(optimizer.values(), i, symbols_pose, symbols_landmarks);
            }
            if (std::abs(prev_error - curr_error) < 1e-6) {
                std::cout << "Converged at iteration " << i << "\n";
                break;
            }
        }
        return optimizer.values();
    }

    static gtsam::Pose3 averagePoses(const std::vector<gtsam::Pose3> &poses, int maxIter = 10) {
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

    static gtsam::Point3 averagePoints(const std::vector<gtsam::Point3> &points) {
        if (points.empty()) throw std::runtime_error("Empty point vector");
        gtsam::Point3 sum(0, 0, 0);
        for (const auto &pt : points) sum += pt;
        return sum / points.size();
    }

    static gtsam::Point3 get_variance(const std::vector<gtsam::Point3> &points) {
        const gtsam::Point3 mean = SFMMvpHelper::averagePoints(points);
        gtsam::Point3 var(0, 0, 0);
        for (const gtsam::Point3 &pt : points) {
            var += (mean - pt).array().square().matrix();
        }
        return var / points.size();
    }
};

TEST(SFMMvp, sfm_building_manual_global) {
    const std::vector<int> indices{60, 80, 100, 120, 140};
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint img_pt_first = survey.getImagePoint(indices.front());

    // const size_t img_width = img_pt_first.width, img_height =
    // img_pt_first.height;
    const double fx = img_pt_first.fx, fy = img_pt_first.fy;
    const double cx = img_pt_first.cx, cy = img_pt_first.cy;
    gtsam::Cal3_S2::shared_ptr K = boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0, cx, cy);
    Eigen::Matrix<double, 5, 1> D =
        (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1, SymphonyLakeCamParams::k2,
         SymphonyLakeCamParams::p1, SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
            .finished();
    cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K->fx(), 0, K->px(), 0, K->fy(), K->py(), 0, 0, 1);
    cv::Mat D_mat = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    // let world be the first boat base recorded. T_world_camera0 = earth_from_boat0
    // * boat_from_camera
    Eigen::Isometry3d earth_from_boat0 = DataParser::get_world_from_boat(img_pt_first);
    Eigen::Isometry3d world_from_boat0;
    world_from_boat0.linear() = earth_from_boat0.linear();
    Eigen::Isometry3d T_world_camera0 =
        world_from_boat0 * DataParser::get_boat_from_camera(img_pt_first);
    // StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
    //                         gtsam::Pose3(T_world_camera0.matrix()));
    Frontend frontend(Frontend::ExtractorType::SIFT, Frontend::MatcherType::KNN);

    gtsam::Values initial_estimate_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 0.04, 0.04, 0.09, 0.01, 0.01, 0.01;
    gtsam::Vector3 gps_sigmas;
    gps_sigmas << 2.1, 2.1, 0.1;
    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    gtsam::noiseModel::Diagonal::shared_ptr gps_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gps_sigmas);

    // populate frames and cam_poses
    std::vector<Frame> frames;
    FrameId id = 0;
    std::unordered_map<FrameId, gtsam::Pose3> cam_pose;
    std::vector<Eigen::Isometry3d> cam_isometries;
    for (const int &idx : indices) {
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        cv::Mat img_undistorted;
        cv::undistort(img, img_undistorted, K_mat, D_mat);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);

        Eigen::Isometry3d world_from_boat = DataParser::get_world_from_boat(img_pt);
        world_from_boat.translation() -= earth_from_boat0.translation();
        Eigen::Isometry3d T_world_cam = world_from_boat * DataParser::get_boat_from_camera(img_pt);
        gtsam::Pose3 T_world_cam_gtsam(T_world_cam.matrix());
        cam_pose.emplace(id, T_world_cam_gtsam);
        cam_isometries.push_back(T_world_cam);

        // graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', id),
        //                                                         T_world_cam_gtsam, pose_noise);

        Frame frame{id, img_undistorted, K, gtsam::Pose3(T_world_cam.matrix())};

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> kpts_descs =
            frontend.extract_features(img_undistorted);
        KeypointsCV kpts;
        for (const cv::KeyPoint &kpt : kpts_descs.first) {
            kpts.push_back(kpt.pt);
        }
        frame.add_keypoints(kpts);
        frame.assign_descriptors(kpts_descs.second);

        frames.push_back(frame);

        id++;
    }

    geometry::viz_scene(cam_isometries, std::vector<Eigen::Vector3d>());

    // populate feature_tracks and lmk_id_map
    FeatureTracks feature_tracks;
    FrameLandmarkIdMap lmk_id_map;
    LandmarkId lmk_id = 0;
    for (size_t i = 0; i < indices.size() - 1; i++) {
        std::vector<cv::DMatch> matches =
            frontend.compute_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
        // frontend.enforce_bijective_matches(matches);
        frontend.enforce_bijective_buffer_matches(matches);
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
        for (const std::pair<FrameId, KeypointCV> &feat_track : feature_track.obs_) {
            initial_estimate_.insert_or_assign(symbol_lmk, gtsam::Point3(0, 0, 0));
            graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                gtsam::Point2(feat_track.second.x, feat_track.second.y), landmark_noise,
                gtsam::Symbol('x', feat_track.first), symbol_lmk, K);

            feat_cam_poses.push_back(cam_pose[feat_track.first]);
            feat_kpts.push_back(gtsam::Point2(feat_track.second.x, feat_track.second.y));
        }
        gtsam::Point3 p_wrld_kpt;
        bool triangulate_success =
            SFMMvpHelper::attempt_triangulate(feat_cam_poses, feat_kpts, K, p_wrld_kpt);
        if (triangulate_success) {
            initial_estimate_.insert_or_assign(symbol_lmk, p_wrld_kpt);
        }
    }
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', 0), cam_pose.at(0),
                                                            prior_pose_noise);
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
        initial_estimate_.insert_or_assign(key_cam, cam_id_pose.second);
        // initial_estimate_.insert_or_assign(key_cam, gtsam::Pose3(gtsam::Rot3(), p_wrld_cam));
    }

    // initial_estimate_.print("huh");
    SFMMvpHelper::graph_values(initial_estimate_, "Confirmation", symbols_pose, symbols_landmarks);

    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
    params.maxIterations = 1;          // We'll manually step it
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_, params);

    double prev_error = optimizer.error();
    const int num_steps = 5;
    const bool viz_itr = true;
    typedef int epoch;
    std::function<void(const gtsam::Values &, const epoch, const std::vector<gtsam::Symbol> &,
                       const std::vector<gtsam::Symbol> &)>
        graph_itr_debug_func = [&](const gtsam::Values &vals, const epoch iter,
                                   const std::vector<gtsam::Symbol> &symbols_pose,
                                   const std::vector<gtsam::Symbol> &symbols_landmarks) {
            std::cout << "iteration " << iter << " complete!";
            std::string window_name = "Iteration_" + std::to_string(iter);
            SFMMvpHelper::graph_values(vals, window_name, symbols_pose, symbols_landmarks);
        };

    SFMMvpHelper::graph_values(initial_estimate_, "Initial Graph", symbols_pose, symbols_landmarks);
    for (int i = 0; i < num_steps; i++) {
        optimizer.iterate();
        double curr_error = optimizer.error();

        if (viz_itr) {
            graph_itr_debug_func(optimizer.values(), i, symbols_pose, symbols_landmarks);
        }
        if (std::abs(prev_error - curr_error) < 1e-6) {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
    }
    const gtsam::Values result = optimizer.values();
}

TEST(SFMMvp, sfm_building_manual_incremental) {
    const std::vector<int> indices{60, 80, 100, 120, 140};
    // const std::vector<int> indices = []() {
    //     std::vector<int> tmp;
    //     for (int i = 0; i < 200; i += 10) {
    //         tmp.push_back(i);
    //     }
    //     return tmp;
    // }();
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey &survey = survey_vector.get(0);
    const symphony_lake_dataset::ImagePoint img_pt_first = survey.getImagePoint(indices.front());

    // const size_t img_width = img_pt_first.width, img_height =
    // img_pt_first.height;
    const double fx = img_pt_first.fx, fy = img_pt_first.fy;
    const double cx = img_pt_first.cx, cy = img_pt_first.cy;
    gtsam::Cal3_S2::shared_ptr K = boost::make_shared<gtsam::Cal3_S2>(fx, fy, 0, cx, cy);
    Eigen::Matrix<double, 5, 1> D =
        (Eigen::Matrix<double, 5, 1>() << SymphonyLakeCamParams::k1, SymphonyLakeCamParams::k2,
         SymphonyLakeCamParams::p1, SymphonyLakeCamParams::p2, SymphonyLakeCamParams::k3)
            .finished();
    cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K->fx(), 0, K->px(), 0, K->fy(), K->py(), 0, 0, 1);
    cv::Mat D_mat = (cv::Mat_<double>(5, 1) << D(0, 0), D(1, 0), D(2, 0), D(3, 0), D(4, 0));

    // let world be the first boat base recorded. T_world_camera0 = earth_from_boat0
    Eigen::Isometry3d earth_from_boat0 = DataParser::get_world_from_boat(img_pt_first);
    Eigen::Isometry3d world_from_boat0;
    world_from_boat0.linear() = earth_from_boat0.linear();
    Eigen::Isometry3d T_world_camera0 =
        world_from_boat0 * DataParser::get_boat_from_camera(img_pt_first);
    // StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
    //                         gtsam::Pose3(T_world_camera0.matrix()));
    Frontend frontend(Frontend::ExtractorType::SIFT, Frontend::MatcherType::KNN);

    gtsam::Values initial_estimate_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise =
        gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 0.04, 0.04, 0.09, 2.1, 2.1, 0.1;
    gtsam::Vector3 gps_sigmas;
    gps_sigmas << 2.1, 2.1, 0.1;
    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    gtsam::noiseModel::Diagonal::shared_ptr gps_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gps_sigmas);

    // populate frames and cam_poses
    std::vector<Frame> frames;
    FrameId id = 0;
    std::unordered_map<FrameId, gtsam::Pose3> cam_pose;
    std::vector<Eigen::Isometry3d> cam_isometries;
    for (const int &idx : indices) {
        const cv::Mat img = survey.loadImageByImageIndex(idx);
        cv::Mat img_undistorted;
        cv::undistort(img, img_undistorted, K_mat, D_mat);
        const symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(idx);

        Eigen::Isometry3d world_from_boat = DataParser::get_world_from_boat(img_pt);
        world_from_boat.translation() -= earth_from_boat0.translation();
        Eigen::Isometry3d T_world_cam = world_from_boat * DataParser::get_boat_from_camera(img_pt);
        gtsam::Pose3 T_world_cam_gtsam(T_world_cam.matrix());
        cam_pose.emplace(id, T_world_cam_gtsam);
        cam_isometries.push_back(T_world_cam);

        Frame frame(id, img_undistorted, K, gtsam::Pose3(T_world_cam.matrix()));

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> kpts_descs =
            frontend.extract_features(img_undistorted);
        KeypointsCV kpts;
        for (const cv::KeyPoint &kpt : kpts_descs.first) {
            kpts.push_back(kpt.pt);
        }
        frame.add_keypoints(kpts);
        frame.assign_descriptors(kpts_descs.second);

        frames.push_back(frame);

        id++;
    }

    geometry::viz_scene(cam_isometries, std::vector<Eigen::Vector3d>());

    // populate feature_tracks and lmk_id_map
    FeatureTracks feature_tracks;
    FrameLandmarkIdMap lmk_id_map;
    LandmarkId lmk_id = 0;
    for (size_t i = 0; i < indices.size() - 1; i++) {
        std::vector<cv::DMatch> matches =
            frontend.compute_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
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

    // triangulate all of the points

    std::unordered_map<LandmarkId, gtsam::Point3> lmk_triangulated_map;
    std::vector<gtsam::Point3> triangulated_lmks;
    for (const std::pair<LandmarkId, FeatureTrack> lmk_feat : feature_tracks) {
        std::vector<gtsam::Pose3> feat_cam_poses;
        std::vector<gtsam::Point2> feat_kpts;
        LandmarkId lmk_id = lmk_feat.first;
        FeatureTrack feature_track = lmk_feat.second;
        gtsam::Point3 p_wrld_kpt(0, 0, 0);
        for (const std::pair<FrameId, KeypointCV> &feat_track : feature_track.obs_) {
            feat_cam_poses.push_back(cam_pose[feat_track.first]);
            feat_kpts.push_back(gtsam::Point2(feat_track.second.x, feat_track.second.y));
        }
        bool triangulate_success =
            SFMMvpHelper::attempt_triangulate(feat_cam_poses, feat_kpts, K, p_wrld_kpt);
        if (triangulate_success) {
            lmk_triangulated_map.emplace(lmk_id, p_wrld_kpt);
            triangulated_lmks.push_back(p_wrld_kpt);
        } else {
            continue;
        }
    }
    // filter points
    geometry::viz_scene(std::vector<Eigen::Isometry3d>(), triangulated_lmks,
                        cv::viz::Color::black(), true, true, "Unfiltered points");
    const gtsam::Point3 variance_pts = SFMMvpHelper::get_variance(triangulated_lmks);
    const gtsam::Point3 std_dev_pts = variance_pts.array().sqrt().matrix();
    const gtsam::Point3 mean_pts = SFMMvpHelper::averagePoints(triangulated_lmks);
    std::unordered_map<LandmarkId, gtsam::Point3> lmk_triangulated_map_filtered;
    std::vector<gtsam::Point3> filtered_points;
    std::cout << "original var " << variance_pts << std::endl;
    for (const std::pair<LandmarkId, gtsam::Point3> lmk_id_pt : lmk_triangulated_map) {
        const gtsam::Point3 dist_mean = (lmk_id_pt.second - mean_pts).array().abs().matrix();
        if ((dist_mean.array() <= (3 * std_dev_pts).array()).all()) {
            lmk_triangulated_map_filtered.emplace(lmk_id_pt);
            filtered_points.push_back(lmk_id_pt.second);
        }
    }
    std::cout << "filtered variance " << SFMMvpHelper::get_variance(filtered_points) << std::endl;
    geometry::viz_scene(std::vector<Eigen::Isometry3d>(), filtered_points, cv::viz::Color::black(),
                        true, true, "Unfiltered points");

    // add filtered points to graph
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Pose3>> symbols_poses_values_iter;
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Point3>> symbols_landmarks_values_iter;
    std::vector<gtsam::Symbol> symbols_pose;
    std::vector<gtsam::Symbol> symbols_landmarks;
    for (const std::pair<LandmarkId, gtsam::Point3> lmk_id_pt : lmk_triangulated_map_filtered) {
        LandmarkId lmk_id = lmk_id_pt.first;
        const gtsam::Point3 p_wrld_kpt = lmk_id_pt.second;
        FeatureTrack feature_track = feature_tracks.at(lmk_id);
        const gtsam::Symbol symbol_lmk('l', lmk_id);
        for (const std::pair<FrameId, KeypointCV> &feat_track : feature_track.obs_) {
            initial_estimate_.insert_or_assign(symbol_lmk, p_wrld_kpt);
            symbols_landmarks_values_iter.emplace(symbol_lmk,
                                                  std::vector<gtsam::Point3>{p_wrld_kpt});
            graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                gtsam::Point2(feat_track.second.x, feat_track.second.y), landmark_noise,
                gtsam::Symbol('x', feat_track.first), symbol_lmk, K);
        }
    }
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', 0), cam_pose.at(0),
                                                            prior_pose_noise);
    initial_estimate_.insert_or_assign(gtsam::Symbol('x', 0), cam_pose.at(0));
    symbols_pose.push_back(gtsam::Symbol('x', 0));
    symbols_poses_values_iter.emplace(gtsam::Symbol('x', 0),
                                      std::vector<gtsam::Pose3>{cam_pose.at(0)});
    for (const std::pair<FrameId, gtsam::Pose3> cam_id_pose : cam_pose) {
        if (cam_id_pose.first == 0) {
            continue;
        }
        const gtsam::Symbol key_cam('x', cam_id_pose.first);
        symbols_pose.push_back(key_cam);
        gtsam::Point3 p_wrld_cam = cam_id_pose.second.translation();
        graph_.emplace_shared<gtsam::GPSFactor>(key_cam, p_wrld_cam, gps_noise);
        initial_estimate_.insert_or_assign(key_cam, cam_id_pose.second);
        symbols_poses_values_iter.emplace(key_cam, std::vector<gtsam::Pose3>{cam_id_pose.second});
        // initial_estimate_.insert_or_assign(key_cam, gtsam::Pose3(gtsam::Rot3(), p_wrld_cam));
    }

    // SFMMvpHelper::graph_values(initial_estimate_, "Confirmation", symbols_pose,
    // symbols_landmarks);

    // do local optimizations and add to iter cache
    // const int window = 2;
    std::cout << "length of cam_poses: " << cam_pose.size() << std::endl;
    for (const auto &id_pose : cam_pose) {
        std::cout << "id: " << id_pose.first << "\tpose: " << id_pose.second << std::endl;
    }
    for (size_t i = 0; i < indices.size() - 1; i++) {
        std::cout << "Local optimization " << i << std::endl;
        gtsam::Values local_estimate_;
        gtsam::NonlinearFactorGraph local_graph_;

        std::vector<gtsam::Symbol> poses{gtsam::Symbol('x', i), gtsam::Symbol('x', i + 1)};
        std::vector<gtsam::Symbol> lmks;

        local_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            gtsam::PriorFactor<gtsam::Pose3>(poses[0], cam_pose.at(i), prior_pose_noise));
        std::cout << "fuck" << std::endl;
        local_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            gtsam::PriorFactor<gtsam::Pose3>(poses[1], cam_pose.at(i + 1), prior_pose_noise));
        std::cout << "mega fuck" << std::endl;
        local_estimate_.insert_or_assign(poses[0], cam_pose.at(i));
        local_estimate_.insert_or_assign(poses[1], cam_pose.at(i + 1));

        std::vector<cv::DMatch> matches =
            frontend.compute_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
        frontend.enforce_bijective_matches(matches);
        std::vector<gtsam::Pose3> feat_cam_poses{cam_pose.at(i), cam_pose.at(i + 1)};

        std::vector<Eigen::Isometry3d> viz_poses{Eigen::Isometry3d(feat_cam_poses[0].matrix()),
                                                 Eigen::Isometry3d(feat_cam_poses[1].matrix())};
        std::vector<Eigen::Vector3d> viz_lmks;
        for (const cv::DMatch match : matches) {
            std::vector<gtsam::Point2> feat_kpts;
            const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
            const KeypointCV kpt_cam1 = frames[i + 1].get_keypoints()[match.trainIdx];
            feat_kpts.emplace_back(kpt_cam0.x, kpt_cam0.y);
            feat_kpts.emplace_back(kpt_cam1.x, kpt_cam1.y);

            auto key = std::make_pair(frames[i].id_, kpt_cam0);
            if (lmk_id_map.find(key) != lmk_id_map.end()) {
                auto id = lmk_id_map.at(key);
                if (lmk_triangulated_map_filtered.find(id) == lmk_triangulated_map_filtered.end()) {
                    continue;
                }
                std::cout << "good" << std::endl;
                // do nothing
                // feature_tracks.at(id).obs_.emplace_back(i, kpt_cam0);
                // feature_tracks.at(id).obs_.emplace_back(i + 1, kpt_cam1);
            } else {
                std::cerr << "this shouldn't happen right?" << std::endl;
                FeatureTrack feature_track(frames[i].id_, kpt_cam0);
                feature_track.obs_.emplace_back(frames[i + 1].id_, kpt_cam1);
                feature_tracks.emplace(lmk_id, feature_track);
                lmk_id_map.emplace(key, lmk_id);
                lmk_id++;
            }
            std::cout << "oog" << std::endl;
            const gtsam::Symbol symbol_lmk =
                gtsam::Symbol('l', lmk_id_map.at(std::make_pair(frames[i].id_, kpt_cam0)));
            // if (gtsam::Symbol('l', lmk_id_map.at(std::make_pair(frames[i].id_, kpt_cam0))) !=
            //     gtsam::Symbol('l', lmk_id_map.at(std::make_pair(frames[i].id_, kpt_cam1)))) {
            //     std::cerr << "UH OH" << std::endl;
            // } else {
            //     std::cout << "cool" << std::endl;
            // }
            lmks.push_back(symbol_lmk);
            local_graph_
                .emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                    feat_kpts[0], landmark_noise, poses[0], symbol_lmk, K);
            local_graph_
                .emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                    feat_kpts[1], landmark_noise, poses[1], symbol_lmk, K);

            gtsam::Point3 p_wrld_kpt;
            bool triangulate_success =
                SFMMvpHelper::attempt_triangulate(feat_cam_poses, feat_kpts, K, p_wrld_kpt);
            if (triangulate_success) {
                local_estimate_.insert_or_assign(symbol_lmk, p_wrld_kpt);
                viz_lmks.push_back(p_wrld_kpt);
            }
            if (symbols_landmarks_values_iter.find(symbol_lmk) !=
                symbols_landmarks_values_iter.end()) {
                symbols_landmarks_values_iter[symbol_lmk].push_back(p_wrld_kpt);
            } else {
                symbols_landmarks_values_iter.emplace(symbol_lmk,
                                                      std::vector<gtsam::Point3>{p_wrld_kpt});
            }
        }
        std::cout << "setup complete!" << std::endl;
        // geometry::viz_scene(viz_poses, viz_lmks, true, true,
        //                     "Local Optimization " + std::to_string(i));

        const gtsam::Values symbols_result_local = SFMMvpHelper::optimize_graph(
            local_graph_, local_estimate_, symbols_pose, symbols_landmarks, false);

        for (const gtsam::Symbol &symbol_pose : poses) {
            const gtsam::Pose3 T_wrld_cam = symbols_result_local.at<gtsam::Pose3>(symbol_pose);
            symbols_poses_values_iter.at(symbol_pose).push_back(T_wrld_cam);
        }
        for (const gtsam::Symbol &symbol_lmk : lmks) {
            const gtsam::Point3 p_wrld_lmk = symbols_result_local.at<gtsam::Point3>(symbol_lmk);
            symbols_landmarks_values_iter.at(symbol_lmk).push_back(p_wrld_lmk);
        }
    }

    std::cout << "\nLocal Optimizations Complete!\n" << std::endl;

    for (std::pair<gtsam::Symbol, std::vector<gtsam::Pose3>> sym_pose : symbols_poses_values_iter) {
        initial_estimate_.insert_or_assign(sym_pose.first,
                                           SFMMvpHelper::averagePoses(sym_pose.second));
    }
    for (std::pair<gtsam::Symbol, std::vector<gtsam::Point3>> sym_lmk :
         symbols_landmarks_values_iter) {
        initial_estimate_.insert_or_assign(sym_lmk.first,
                                           SFMMvpHelper::averagePoints(sym_lmk.second));
    }

    // do global optimization
    const gtsam::Values result =
        SFMMvpHelper::optimize_graph(graph_, initial_estimate_, symbols_pose, symbols_landmarks);
    std::cout << "about to visualize result" << std::endl;
    SFMMvpHelper::graph_values(result, "Result", symbols_pose, symbols_landmarks);
}

}  // namespace robot::experimental::learn_descriptors