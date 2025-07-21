#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/geometry/camera.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/backend.hh"
#include "experimental/learn_descriptors/camera_calibration.hh"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "experimental/learn_descriptors/image_point_four_seasons.hh"
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
#include "opencv2/opencv.hpp"
#include "visualization/opencv/opencv_viz.hh"

std::optional<gtsam::Point3> attempt_triangulate(const std::vector<gtsam::Pose3> &cam_poses,
                                                 const std::vector<gtsam::Point2> &cam_obs,
                                                 gtsam::Cal3_S2::shared_ptr K,
                                                 const double max_reproj_error = 2.0,
                                                 const bool verbose = true) {
    gtsam::Point3 p_lmk_in_world;
    if (cam_poses.size() > 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            p_lmk_in_world = gtsam::triangulatePoint3(
                cam_poses, K, gtsam::Point2Vector(cam_obs.begin(), cam_obs.end()));

        } catch (const std::exception &e) {
            // Handle the exception gracefully by logging and retaining the previous
            // estimate or discard
            if (verbose) {
                std::cerr << "[attempt_triangulate] failed. Likely cheirality exception: "
                          << e.what() << ". Discarding involved keypoints." << std::endl;
            }
            return std::nullopt;
        }
    } else {
        return std::nullopt;
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
                std::cerr << "[attempt_triangulate] point behind camera " << i
                          << " (z=" << p_cam.z() << ")\n";
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
                    std::cerr << "[attempt_triangulate] reprojection error too large on view " << i
                              << " (" << err << " px)\n";
                }
                return std::nullopt;
            }
        }
    }
    return p_lmk_in_world;
}

void graph_values(const gtsam::Values &values, const std::string &window_name,
                  const std::vector<gtsam::Symbol> &symbols_pose,
                  const std::vector<gtsam::Symbol> &symbols_landmarks) {
    std::vector<robot::visualization::VizPose> final_poses;
    std::vector<robot::visualization::VizPoint> final_lmks;
    for (const gtsam::Symbol &symbol_pose : symbols_pose) {
        final_poses.emplace_back(Eigen::Isometry3d(values.at<gtsam::Pose3>(symbol_pose).matrix()),
                                 symbol_pose.string());
    }
    for (const gtsam::Symbol &symbol_lmk : symbols_landmarks) {
        if (!values.exists(symbol_lmk)) {
            std::cout << "WTF " << symbol_lmk << std::endl;
        }
        final_lmks.emplace_back(values.at<gtsam::Point3>(symbol_lmk), symbol_lmk.string());
    }
    std::cout << "About to viz gtsam::Values with " << values.size() << " variables." << std::endl;
    robot::visualization::viz_scene(final_poses, final_lmks, cv::viz::Color::brown(), true, true,
                                    window_name);
}

gtsam::Values optimize_graph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &values,
                             const std::vector<gtsam::Symbol> &symbols_pose,
                             const std::vector<gtsam::Symbol> &symbols_landmarks,
                             const int num_epochs = 5, bool viz_itr = false) {
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
    params.maxIterations = 1;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);

    double prev_error = optimizer.error();
    typedef int epoch;
    std::function<void(const gtsam::Values &, const epoch, const std::vector<gtsam::Symbol> &,
                       const std::vector<gtsam::Symbol> &)>
        graph_itr_debug_func = [&](const gtsam::Values &vals, const epoch iter,
                                   const std::vector<gtsam::Symbol> &symbols_pose,
                                   const std::vector<gtsam::Symbol> &symbols_landmarks) {
            std::cout << "iteration " << iter << " complete!";
            std::string window_name = "Iteration_" + std::to_string(iter);
            graph_values(vals, window_name, symbols_pose, symbols_landmarks);
        };

    for (int i = 0; i < num_epochs; i++) {
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

double rotation_error(const Eigen::Isometry3d &T_est, const Eigen::Isometry3d &T_gt) {
    Eigen::Matrix3d R_err = T_gt.rotation().transpose() * T_est.rotation();

    // 2. Compute trace and clamp to [-1,1] for numerical safety
    double tr = R_err.trace();
    double cos_theta = std::min(1.0, std::max(-1.0, (tr - 1.0) / 2.0));

    // 3. Recover angle (in radians)
    return std::acos(cos_theta);
}

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

    FrontendParams params{FrontendParams::ExtractorType::SIFT, FrontendParams::MatcherType::KNN,
                          true, false};
    Frontend frontend(params);

    std::vector<size_t> idx_img_pts;
    for (size_t i = 633; i < 646; i += 2) {
        frontend.add_image(
            ImageAndPoint{parser.load_image(i),
                          std::make_shared<ImagePointFourSeasons>(parser.get_image_point(i))});
        idx_img_pts.push_back(i);
    }
    // for (size_t i = 581; i < 750; i += 5) {
    //     frontend.add_image(
    //         ImageAndPoint{parser.load_image(i),
    //                       std::make_shared<ImagePointFourSeasons>(parser.get_image_point(i))});
    //     idx_img_pts.push_back(i);
    // }
    frontend.populate_frames();
    frontend.match_frames_and_build_tracks();

    std::cout << "first frame translation: " << *frontend.frames()[0].cam_in_world_initial_guess_
              << std::endl;
    std::cout << "ground truth translation: "
              << frontend.frames()[0].world_from_cam_groundtruth_->translation() << std::endl;

    std::vector<robot::visualization::VizPose> viz_poses_init;
    std::optional<Eigen::Vector3d> first_point;
    std::optional<Eigen::Isometry3d> first_grnd_trth;
    (void)first_point;
    for (const Frame &frame : frontend.frames()) {
        // if (frame.cam_in_world_initial_guess_) {
        //     Eigen::Isometry3d world_from_cam_init;
        //     world_from_cam_init.linear() = frame.world_from_cam_initial_guess_->matrix();
        //     if (!first_point) first_point = *frame.cam_in_world_initial_guess_;
        //     world_from_cam_init.translation() = *frame.cam_in_world_initial_guess_ -
        //     *first_point; std::cout << "world_from_cam_init: " <<
        //     world_from_cam_init.matrix() << std::endl; viz_poses_init.emplace_back(
        //         world_from_cam_init, gtsam::Symbol(Frontend::symbol_pose_char,
        //         frame.id_).string());
        // }
        if (frame.world_from_cam_groundtruth_) {
            std::cout << "adding a ground truth frame to viz!" << std::endl;
            Eigen::Isometry3d w_from_cam_grnd_trth(frame.world_from_cam_groundtruth_->matrix());
            if (!first_grnd_trth) first_grnd_trth = w_from_cam_grnd_trth;
            w_from_cam_grnd_trth.translation() -= first_grnd_trth->translation();
            viz_poses_init.emplace_back(w_from_cam_grnd_trth,
                                        "x_grnd_" + std::to_string(frame.id_));
        };
    }

    std::cout << "visualizing " << viz_poses_init.size() << " poses" << std::endl;
    robot::visualization::viz_scene(viz_poses_init, std::vector<robot::visualization::VizPoint>(),
                                    cv::viz::Color::brown(), true, true,
                                    "Frontend initial guesses");

    // ############# BACKEND ###############
    constexpr bool use_grndtrth_only = false;
    gtsam::Values initial_estimate_;
    gtsam::NonlinearFactorGraph graph_;

    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise = gtsam::noiseModel::Isotropic::Sigma(
        2, 1.0);  // should probably mess with this, could also use essential matrix values to guide
                  // this potentially
    gtsam::Vector6 prior_sigmas;
    prior_sigmas << 0.04, 0.04, 0.09, 2.1, 2.1, 0.1;
    gtsam::Vector3 gps_sigmas_fallback;
    gps_sigmas_fallback << 1.5, 1.5, 1.5;
    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
    gtsam::noiseModel::Diagonal::shared_ptr gps_noise =
        gtsam::noiseModel::Diagonal::Sigmas(gps_sigmas_fallback);

    // add gps factors
    std::vector<Frame> &frames = frontend.frames();
    frames[0].world_from_cam_initial_guess_ =
        frames[0].world_from_cam_groundtruth_->rotation();  // make sure first idx has grnd trth

    // // let's examine the rotations to the first thingy
    // std::vector<robot::visualization::VizPose> viz_5_pt_rot;
    // for (const auto &[frame_id, rotation] : frames[0].frame_from_other_frames_) {
    //     viz_5_pt_rot.emplace_back(
    //         Eigen::Isometry3d(gtsam::Pose3(rotation, gtsam::Point3()).matrix()),
    //         "rot_to_" + std::to_string(frame_id));
    // }
    // robot::visualization::viz_scene(viz_5_pt_rot, std::vector<robot::visualization::VizPoint>(),
    //                                 cv::viz::Color::brown(), true, true,
    //                                 "Frontend initial guesses");

    Backend::populate_rotation_estimate(frames);
    std::vector<gtsam::Symbol> symbols_pose;
    std::unordered_map<size_t, gtsam::Pose3> world_from_cam_initial_guess;
    std::optional<std::vector<Eigen::Vector3d>> interpolated_cam_in_w =
        frontend.interpolated_initial_translations();
    ROBOT_CHECK(interpolated_cam_in_w, interpolated_cam_in_w);
    // std::vector<robot::visualization::VizPoint> viz_pts_interpolated;
    // for (size_t i = 0; i < interpolated_cam_in_w->size(); i++) {
    //     viz_pts_interpolated.emplace_back(
    //         (*interpolated_cam_in_w)[i] - interpolated_cam_in_w->front(),
    //         "pt_" + std::to_string(i));
    //     std::cout << "pt_" + std::to_string(i) << ": " << (*interpolated_cam_in_w)[i] <<
    //     std::endl;
    // }
    // robot::visualization::viz_scene(std::vector<robot::visualization::VizPose>(),
    //                                 viz_pts_interpolated, cv::viz::Color::brown(), true, true,
    //                                 "Initial guesses in backend");
    Eigen::Vector3d cam0_in_w = interpolated_cam_in_w->front();
    std::optional<Eigen::Vector3d> grndtrth_cam0_in_w;
    std::vector<size_t> idx_grndtrth_frame;
    std::vector<robot::visualization::VizPose> viz_pose;
    auto noise_tight_prior = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0, 0, 0,  // rotation stdev in radians
         1e-6, 1e-6, 1e-6              // translation stdev in meters
         )
            .finished());
    const gtsam::Pose3 w_from_cam0_init_estimate(*frames[0].world_from_cam_initial_guess_,
                                                 gtsam::Point3());
    const gtsam::Symbol symbol_cam0(Backend::symbol_char_pose, 0);
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
        symbol_cam0, w_from_cam0_init_estimate,
        noise_tight_prior));  // currently assuming that we have groundtruth on the first pose
    initial_estimate_.insert(symbol_cam0, w_from_cam0_init_estimate);
    std::string str_pose0 =
        std::string("x_") + (frames[0].world_from_cam_groundtruth_ ? "g_" : "") +
        (!frames[0].world_from_cam_initial_guess_ ? "norot_" : "") + std::to_string(frames[0].id_);
    viz_pose.emplace_back(Eigen::Isometry3d(w_from_cam0_init_estimate.matrix()), str_pose0);
    symbols_pose.push_back(symbol_cam0);
    for (size_t i = 1; i < frames.size(); i++) {
        const Frame &frame = frames[i];
        std::optional<gtsam::Pose3> world_from_cam_estimate;
        if (use_grndtrth_only && !frame.world_from_cam_groundtruth_) {
            continue;
        } else if (use_grndtrth_only && frame.world_from_cam_groundtruth_) {
            if (!grndtrth_cam0_in_w)
                grndtrth_cam0_in_w = frame.world_from_cam_groundtruth_->translation();
            world_from_cam_estimate =
                gtsam::Pose3(frame.world_from_cam_groundtruth_->rotation(),
                             gtsam::Point3(frame.world_from_cam_groundtruth_->translation() -
                                           *grndtrth_cam0_in_w));
            idx_grndtrth_frame.push_back(i);
            std::cout << "populating groundtruth pose: " << world_from_cam_estimate->translation()
                      << std::endl;
        } else {
            world_from_cam_estimate = gtsam::Pose3(frame.world_from_cam_initial_guess_
                                                       ? *frame.world_from_cam_initial_guess_
                                                       : gtsam::Rot3::Identity(),
                                                   (*interpolated_cam_in_w)[i] - cam0_in_w);
        }
        std::cout << "world_from_cam_estimate_" << i << ": " << *world_from_cam_estimate
                  << std::endl;
        world_from_cam_initial_guess[frame.id_] = *world_from_cam_estimate;
        const gtsam::Symbol cam_symbol(Backend::symbol_char_pose, frame.id_);
        initial_estimate_.insert(cam_symbol, *world_from_cam_estimate);
        if (frame.cam_in_world_initial_guess_) {
            if (frame.translation_covariance_in_cam_) {
                std::cout << "translation covariance: " << *frame.translation_covariance_in_cam_
                          << std::endl;
            }
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
                // frame.translation_covariance_in_cam_
                false ? gtsam::Vector3(std::sqrt((*frame.translation_covariance_in_cam_)(0, 0)),
                                       std::sqrt((*frame.translation_covariance_in_cam_)(1, 1)),
                                       std::sqrt((*frame.translation_covariance_in_cam_)(2, 2)))
                      : gps_sigmas_fallback);
            graph_.add(gtsam::GPSFactor(cam_symbol, *frame.cam_in_world_initial_guess_, gps_noise));
            std::cout << "adding gps factor!" << std::endl;
        }
        std::string str_pose = std::string("x_") + (frame.world_from_cam_groundtruth_ ? "g_" : "") +
                               (!frame.world_from_cam_initial_guess_ ? "norot_" : "") +
                               std::to_string(frame.id_);
        viz_pose.emplace_back(Eigen::Isometry3d(world_from_cam_estimate->matrix()), str_pose);
        symbols_pose.push_back(cam_symbol);
    }

    // add landmarks to graph
    std::vector<gtsam::Symbol> symbols_lmks;
    std::vector<robot::visualization::VizPoint> viz_pts;
    const FeatureTracks feature_tracks = frontend.feature_tracks();
    for (size_t i = 0; i < feature_tracks.size(); i++) {
        std::vector<gtsam::Pose3> world_from_lmk_cams;
        std::vector<gtsam::Point2> lmk_observations;
        for (const auto &[frame_id, keypoint_cv] : feature_tracks[i].obs_) {
            world_from_lmk_cams.push_back(world_from_cam_initial_guess[frame_id]);
            lmk_observations.emplace_back(keypoint_cv.x, keypoint_cv.y);
        }
        std::optional<gtsam::Point3> landmark_estimate =
            attempt_triangulate(world_from_lmk_cams, lmk_observations, frames[0].K_, 2.0,
                                false);  // all K are the same for now...
        if (landmark_estimate) {
            const gtsam::Symbol lmk_symbol(Backend::symbol_char_landmark, i);
            for (const auto &[frame_id, keypoint_cv] : feature_tracks[i].obs_) {
                graph_.add(gtsam::GenericProjectionFactor(
                    gtsam::Point2(keypoint_cv.x, keypoint_cv.y), landmark_noise,
                    gtsam::Symbol(Backend::symbol_char_pose, frame_id), lmk_symbol, frames[0].K_));
            }
            initial_estimate_.insert(lmk_symbol, *landmark_estimate);
            symbols_lmks.push_back(lmk_symbol);
            viz_pts.emplace_back(*landmark_estimate, "lmk_" + std::to_string(i));
        }
    }
    robot::visualization::viz_scene(viz_pose, viz_pts, cv::viz::Color::brown(), true, true,
                                    "Initial guesses in backend");

    // do global optimization
    const gtsam::Values result =
        optimize_graph(graph_, initial_estimate_, symbols_pose, symbols_lmks, 10);

    // calculate ATE (Absolute Trajectory Error) average (RMSE) to reference
    double sum_traj_error = 0;
    double sum_rot_error = 0;
    // for (size_t i = 0; i < idx_grndtrth_frame.size(); i++) {
    for (const size_t i_grndtrth_frame : idx_grndtrth_frame) {
        const Frame &frame = frames[i_grndtrth_frame];
        const gtsam::Pose3 traj_pose =
            result.at<gtsam::Pose3>(gtsam::Symbol(Backend::symbol_char_pose, frame.id_));
        sum_traj_error += std::pow(
            (frame.world_from_cam_groundtruth_->translation() - traj_pose.translation()).norm(), 2);
        sum_rot_error +=
            rotation_error(Eigen::Isometry3d(frame.world_from_cam_groundtruth_->matrix()),
                           Eigen::Isometry3d(traj_pose.matrix()));
    }
    std::cout << "sum_rot_error: " << sum_rot_error << std::endl;
    double rmse_ate = std::sqrt(sum_traj_error / symbols_pose.size());
    double rmse_rot = std::sqrt(sum_rot_error / symbols_pose.size());
    std::cout << "\n\nRMSE_ATE:\t" << rmse_ate << "\nRMSE_ROT:\t" << rmse_rot << std::endl;

    std::cout << "about to visualize result" << std::endl;
    // result.print();
    graph_values(result, "Result", symbols_pose, symbols_lmks);
}