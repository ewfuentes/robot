#include <cmath>
#include <functional>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <sstream>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/geometry/camera.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/camera_calibration.hh"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
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
#include "visualization/opencv/opencv_viz.hh"

int main(int argc, const char **argv) {
    using namespace robot::experimental::learn_descriptors;
    // clang-format off
    cxxopts::Options options("four_seasons_parser_example", "Demonstrate usage of four_seasons_parser");
    options.add_options()
    ("data_dir", "Path to dataset root directory", cxxopts::value<std::string>())
    ("calibration_dir", "Path to dataset calibration directory", cxxopts::value<std::string>())
    ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    const auto check_required = [&](const std::string &opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    check_required("data_dir");
    check_required("calibration_dir");

    const std::filesystem::path path_data = args["data_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();
    FourSeasonsParser parser(path_data, path_calibration);

    std::cout << "refactor_sfm_mvp!" << std::endl;
    // const std::vector<int> indices{581, 609,
    //                                633};  // indices with all data fields on neighborhood_5_train
    // const std::vector<int> indices{581, 593, 609, 621, 633, 636, 639, 642};  // indices with
    // get indices with fully populated img_pt containing full gps data and reference
    // const std::vector<int> indices = [&parser]() -> std::vector<int> {
    //     std::vector<int> tmp;
    //     for (size_t i = 0; i < parser.num_images(); i++) {
    //         const ImagePointFourSeasons img_pt = parser.get_image_point(i);
    //         if (img_pt.AS_w_from_gnss_cam && img_pt.gps_gcs && img_pt.gps_gcs &&
    //             img_pt.gps_gcs->uncertainty && img_pt.gps_gcs->altitude)
    //             tmp.push_back(i);
    //     }
    //     return tmp;
    // }();
    constexpr bool visualize = false;
    // all data fields on neighborhood_5_train const std::vector<int> indices = []() {
    //     std::vector<int> tmp;
    //     for (int i = 660; i < 681; i += 10) {
    //         tmp.push_back(i);
    //     }
    //     return tmp;
    // }();
    // FourSeasonsParser parser()
    // const size_t img_width = img_pt_first.width, img_height =
    // img_pt_first.height;

    // StructureFromMotion sfm(Frontend::ExtractorType::SIFT, K, D,
    //                         gtsam::Pose3(T_world_camera0.matrix()));
    FrontendParams params{FrontendParams::ExtractorType::SIFT, FrontendParams::MatcherType::KNN,
                          true, false};
    Frontend frontend(params);

    for (size_t i = 660, i < 1000; i += 5) {
        frontend.add_images(ImageAndPoint{parser.load_image(i), parser.get_image_point(i)});
    }
    frontend.populate_frames();
    frontend.match_frames_and_build_tracks();

    // ############# BACKEND ###############
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

    // add gps factors
    const std::vector<Frame> &frames = frontned.frames();
    graph.add(PriorFactor<Pose3>(Symbol('x', 0), gtsam::Pose3::Identity(),
                                 noiseModel::Constrained::All(6)  // effectively fixes the pose
                                 ));
    initial_estimate_.insert(Symbol('x', 0), gtsam::Pose3::Identity());
    for (size_t i = 1; i < frames.size(); i++) {
        if (frames[i].) graph.add(gtsam::GPSFactor(Symbol('x', i), ));
    }

    // add filtered points to graph
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Pose3>> symbols_poses_values_iter;
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Point3>> symbols_landmarks_values_iter;
    std::vector<gtsam::Symbol> symbols_pose;
    std::vector<gtsam::Symbol> symbols_landmarks;
    for (const auto &[lmk_id, kpt_in_world] : lmk_triangulated_map_filtered) {
        // LandmarkId lmk_id = lmk_id_pt.first;
        // const gtsam::Point3 kpt_in_world = lmk_id_pt.second;
        FeatureTrack feature_track = feature_tracks.at(lmk_id);
        const gtsam::Symbol symbol_lmk('l', lmk_id);
        for (const auto &[frame_id, obs] : feature_track.obs_) {
            initial_estimate_.insert_or_assign(symbol_lmk, kpt_in_world);
            symbols_landmarks.push_back(symbol_lmk);
            symbols_landmarks_values_iter.emplace(symbol_lmk,
                                                  std::vector<gtsam::Point3>{kpt_in_world});
            graph_.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                gtsam::Point2(obs.x, obs.y), landmark_noise, gtsam::Symbol('x', frame_id),
                symbol_lmk, K);
        }
    }

    // add gps factors
    graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        gtsam::Symbol('x', 0), id_to_initial_world_from_cam.at(0), prior_pose_noise);
    initial_estimate_.insert_or_assign(gtsam::Symbol('x', 0), id_to_initial_world_from_cam.at(0));
    symbols_pose.push_back(gtsam::Symbol('x', 0));
    symbols_poses_values_iter.emplace(
        gtsam::Symbol('x', 0), std::vector<gtsam::Pose3>{id_to_initial_world_from_cam.at(0)});
    for (const auto &[frame_id, estimate_world_from_cam] : id_to_initial_world_from_cam) {
        if (frame_id == 0) {
            continue;
        }
        const gtsam::Symbol key_cam('x', frame_id);
        symbols_pose.push_back(key_cam);
        gtsam::Point3 p_cam_in_world = estimate_world_from_cam.translation();
        graph_.emplace_shared<gtsam::GPSFactor>(key_cam, p_cam_in_world, gps_noise);
        initial_estimate_.insert_or_assign(key_cam, estimate_world_from_cam);
        symbols_poses_values_iter.emplace(key_cam,
                                          std::vector<gtsam::Pose3>{estimate_world_from_cam});
        // initial_estimate_.insert_or_assign(key_cam, gtsam::Pose3(gtsam::Rot3(), p_cam_in_world));
    }

    // detail_sfm::graph_values(initial_estimate_, "Confirmation", symbols_pose, symbols_landmarks);
    std::cout << "heart beat 3" << std::endl;

    constexpr bool local_optimizations = false;
    if (local_optimizations) {
        // TODO: do local optimizations between groups of any n nubmer of cameras with >= x number
        // of matches
        // do local optimizations and add to iter cache
        // const int window = 2;
        std::cout << "length of cam_poses: " << id_to_initial_world_from_cam.size() << std::endl;
        for (const auto &[frame_id, world_from_cam] : id_to_initial_world_from_cam) {
            std::cout << "id: " << frame_id << "\tpose: " << world_from_cam << std::endl;
        }
        for (size_t i = 0; i < indices.size() - 1; i++) {
            std::cout << "Local optimization " << i << std::endl;
            gtsam::Values local_estimate_;
            gtsam::NonlinearFactorGraph local_graph_;

            std::vector<gtsam::Symbol> symbols_poses{gtsam::Symbol('x', i),
                                                     gtsam::Symbol('x', i + 1)};
            std::vector<gtsam::Symbol> symbols_lmks;

            local_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                gtsam::PriorFactor<gtsam::Pose3>(
                    symbols_poses[0], id_to_initial_world_from_cam.at(i), prior_pose_noise));
            std::cout << "fuck" << std::endl;
            local_graph_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                gtsam::PriorFactor<gtsam::Pose3>(
                    symbols_poses[1], id_to_initial_world_from_cam.at(i + 1), prior_pose_noise));
            std::cout << "mega fuck" << std::endl;
            local_estimate_.insert_or_assign(symbols_poses[0], id_to_initial_world_from_cam.at(i));
            local_estimate_.insert_or_assign(symbols_poses[1],
                                             id_to_initial_world_from_cam.at(i + 1));

            std::vector<cv::DMatch> matches = frontend.compute_matches(
                frames[i].get_descriptors(), frames[i + 1].get_descriptors());
            // DIAL TO MESS WITH
            frontend.enforce_bijective_matches(matches);
            std::vector<gtsam::Pose3> world_from_cams{id_to_initial_world_from_cam.at(i),
                                                      id_to_initial_world_from_cam.at(i + 1)};

            std::vector<robot::geometry::VizPose> viz_world_from_cams{
                {Eigen::Isometry3d(world_from_cams[0].matrix()), "cam 0"},
                {Eigen::Isometry3d(world_from_cams[1].matrix()), "cam 1"}};
            std::vector<robot::geometry::VizPoint> viz_lmks;
            for (const cv::DMatch match : matches) {
                std::vector<gtsam::Point2> feat_kpts;
                const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
                const KeypointCV kpt_cam1 = frames[i + 1].get_keypoints()[match.trainIdx];
                feat_kpts.emplace_back(kpt_cam0.x, kpt_cam0.y);
                feat_kpts.emplace_back(kpt_cam1.x, kpt_cam1.y);

                const std::pair<FrameId, KeypointCV> key_lmk_id =
                    std::make_pair(frames[i].id_, kpt_cam0);
                if (lmk_id_map.find(key_lmk_id) != lmk_id_map.end()) {
                    LandmarkId match_lmk_id = lmk_id_map.at(key_lmk_id);
                    if (lmk_triangulated_map_filtered.find(match_lmk_id) ==
                        lmk_triangulated_map_filtered.end()) {
                        continue;
                    }
                    std::cout << "good" << std::endl;
                    // do nothing
                    // feature_tracks.at(id).obs_.emplace_back(i, kpt_cam0);
                    // feature_tracks.at(id).obs_.emplace_back(i + 1, kpt_cam1);
                } else {
                    std::cerr << "ERROR: this shouldn't happen right?" << std::endl;
                    FeatureTrack feature_track(frames[i].id_, kpt_cam0);
                    feature_track.obs_.emplace_back(frames[i + 1].id_, kpt_cam1);
                    feature_tracks.emplace(lmk_id, feature_track);
                    lmk_id_map.emplace(key_lmk_id, lmk_id);
                    lmk_id++;
                }
                std::cout << "oog" << std::endl;
                const gtsam::Symbol symbol_lmk = gtsam::Symbol('l', lmk_id_map.at(key_lmk_id));
                // if (gtsam::Symbol('l', lmk_id_map.at(std::make_pair(frames[i].id_, kpt_cam0))) !=
                //     gtsam::Symbol('l', lmk_id_map.at(std::make_pair(frames[i].id_, kpt_cam1)))) {
                //     std::cerr << "UH OH" << std::endl;
                // } else {
                //     std::cout << "cool" << std::endl;
                // }
                symbols_lmks.push_back(symbol_lmk);
                local_graph_
                    .emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                        feat_kpts[0], landmark_noise, symbols_poses[0], symbol_lmk, K);
                local_graph_
                    .emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3>>(
                        feat_kpts[1], landmark_noise, symbols_poses[1], symbol_lmk, K);

                std::optional<gtsam::Point3> kpt_in_world =
                    detail_sfm::attempt_triangulate(world_from_cams, feat_kpts, K);
                // gtsam::Point3 kpt_in_world;
                // bool triangulate_success =
                //     detail_sfm::attempt_triangulate(world_from_cams, feat_kpts, K, kpt_in_world);
                if (kpt_in_world) {
                    local_estimate_.insert_or_assign(symbol_lmk, *kpt_in_world);
                    viz_lmks.emplace_back(*kpt_in_world);
                    if (symbols_landmarks_values_iter.find(symbol_lmk) !=
                        symbols_landmarks_values_iter.end()) {
                        symbols_landmarks_values_iter[symbol_lmk].push_back(*kpt_in_world);
                    } else {
                        symbols_landmarks_values_iter.emplace(
                            symbol_lmk, std::vector<gtsam::Point3>{*kpt_in_world});
                    }
                }
            }
            std::cout << "setup complete!" << std::endl;
            if (visualize)
                robot::geometry::viz_scene(viz_world_from_cams, viz_lmks, cv::viz::Color::brown(),
                                           true, true, "Local Optimization " + std::to_string(i));

            const gtsam::Values symbols_result_local = detail_sfm::optimize_graph(
                local_graph_, local_estimate_, symbols_pose, symbols_landmarks, false);

            for (const gtsam::Symbol &symbol_pose : symbols_poses) {
                const gtsam::Pose3 T_wrld_cam = symbols_result_local.at<gtsam::Pose3>(symbol_pose);
                symbols_poses_values_iter.at(symbol_pose).push_back(T_wrld_cam);
            }
            for (const gtsam::Symbol &symbol_lmk : symbols_lmks) {
                const gtsam::Point3 p_wrld_lmk = symbols_result_local.at<gtsam::Point3>(symbol_lmk);
                symbols_landmarks_values_iter.at(symbol_lmk).push_back(p_wrld_lmk);
            }
        }

        std::cout << "\nLocal Optimizations Complete!\n" << std::endl;

        for (std::pair<gtsam::Symbol, std::vector<gtsam::Pose3>> sym_pose :
             symbols_poses_values_iter) {
            initial_estimate_.insert_or_assign(sym_pose.first,
                                               detail_sfm::averagePoses(sym_pose.second));
        }
        for (std::pair<gtsam::Symbol, std::vector<gtsam::Point3>> sym_lmk :
             symbols_landmarks_values_iter) {
            initial_estimate_.insert_or_assign(sym_lmk.first,
                                               detail_sfm::averagePoints(sym_lmk.second));
        }
    }

    // do global optimization
    const gtsam::Values result =
        detail_sfm::optimize_graph(graph_, initial_estimate_, symbols_pose, symbols_landmarks, 0);

    // calculate ATE (Absolute Trajectory Error) average (RMSE) to reference
    double sum_traj_error = 0;
    double sum_rot_error = 0;
    for (size_t i = 0; i < symbols_pose.size(); i++) {
        const gtsam::Pose3 traj_pose = result.at<gtsam::Pose3>(symbols_pose[i]);
        sum_traj_error += std::pow(
            (references_world_from_cam[i].translation() - traj_pose.translation()).norm(), 2);
        sum_rot_error += detail_sfm::rotation_error(references_world_from_cam[i],
                                                    Eigen::Isometry3d(traj_pose.matrix()));
    }
    std::cout << "sum_rot_error: " << sum_rot_error << std::endl;
    double rmse_ate = std::sqrt(sum_traj_error / symbols_pose.size());
    double rmse_rot = std::sqrt(sum_rot_error / symbols_pose.size());
    std::cout << "\n\nRMSE_ATE:\t" << rmse_ate << "\nRMSE_ROT:\t" << rmse_rot << std::endl;

    std::cout << "about to visualize result" << std::endl;
    result.print();
    detail_sfm::graph_values(result, "Result", symbols_pose, std::vector<gtsam::Symbol>());
    // detail_sfm::graph_values(result, "Result", symbols_pose, symbols_landmarks);
}