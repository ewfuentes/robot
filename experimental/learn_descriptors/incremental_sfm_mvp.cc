#include <functional>
#include <opencv2/calib3d.hpp>
#include <sstream>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cxxopts.hpp"
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

namespace std {
template <>
struct hash<gtsam::Symbol> {
    size_t operator()(const gtsam::Symbol &s) const { return hash<uint64_t>()(s.key()); }
};
}  // namespace std

namespace detail_sfm {

bool attempt_triangulate(const std::vector<gtsam::Pose3> &cam_poses,
                         const std::vector<gtsam::Point2> &cam_obs, gtsam::Cal3_S2::shared_ptr K,
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
            std::cerr << "attamp_triangulation failed. Likely cheirality exception: " << e.what()
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
    robot::geometry::viz_scene(final_poses, final_lmks, true, true, window_name);
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
            graph_values(vals, window_name, symbols_pose, symbols_landmarks);
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
    const gtsam::Point3 mean = averagePoints(points);
    gtsam::Point3 var(0, 0, 0);
    for (const gtsam::Point3 &pt : points) {
        var += (mean - pt).array().square().matrix();
    }
    return var / points.size();
}
};  // namespace detail_sfm

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

    std::cout << "incremental_sfm_mvp!" << std::endl;
    // const std::vector<int> indices{650, 675, 140, 160, 180};
    const std::vector<int> indices = []() {
        std::vector<int> tmp;
        for (int i = 660; i < 700; i += 3) {
            tmp.push_back(i);
        }
        return tmp;
    }();
    // FourSeasonsParser parser()
    // const size_t img_width = img_pt_first.width, img_height =
    // img_pt_first.height;
    const FourSeasonsParser::CameraCalibrationFisheye cal_parser_left_cam =
        parser.camera_calibration();
    gtsam::Cal3_S2::shared_ptr K =
        boost::make_shared<gtsam::Cal3_S2>(cal_parser_left_cam.fx, cal_parser_left_cam.fy, 0,
                                           cal_parser_left_cam.cx, cal_parser_left_cam.cy);
    cv::Mat K_mat = (cv::Mat_<double>(3, 3) << K->fx(), 0, K->px(), 0, K->fy(), K->py(), 0, 0, 1);
    cv::Mat D_mat = (cv::Mat_<double>(4, 1) << cal_parser_left_cam.k1, cal_parser_left_cam.k2,
                     cal_parser_left_cam.k3, cal_parser_left_cam.k4);

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
    std::unordered_map<FrameId, gtsam::Pose3>
        id_to_initial_world_from_cam;  // these are for initial guesses
    std::vector<Eigen::Isometry3d> references_world_from_cam;
    Eigen::Matrix4d scale_mat_reference = Eigen::Matrix4d::Identity();
    scale_mat_reference(0, 0) = scale_mat_reference(1, 1) = scale_mat_reference(2, 2) =
        parser.gnss_scale();
    for (const int &idx : indices) {
        const ImagePoint img_pt = parser.get_image_point(idx);
        //
        Eigen::Isometry3d world_from_cam;
        if (id == 0) {
            // for now we will use an informative prior based on reference
            world_from_cam = Eigen::Isometry3d((parser.S_from_AS().matrix() * scale_mat_reference *
                                                img_pt.AS_w_from_gnss_cam->matrix())
                                                   .matrix());
        } else if (img_pt.gps_gcs) {
            const Eigen::Vector3d gps_gcs(
                img_pt.gps_gcs->latitude, img_pt.gps_gcs->longitude,
                img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0);
            const Eigen::Vector3d p_gps_in_ECEF = parser.ECEF_from_gcs(gps_gcs);
            const Eigen::Vector4d p_gps_in_ECEF_hom(p_gps_in_ECEF.x(), p_gps_in_ECEF.y(),
                                                    p_gps_in_ECEF.z(), 1.0);
            world_from_cam.translation() =
                ((parser.w_from_gpsw() * parser.e_from_gpsw().inverse()).matrix() *
                 p_gps_in_ECEF_hom)
                    .head<3>();

            // Eigen::
        }
        id_to_initial_world_from_cam.emplace(id, world_from_cam.matrix());

        // populate frame
        cv::Mat img_undistorted = parser.load_image(idx);
        // cv::Mat img = parser.load_image(idx);
        // cv::Mat img_undistorted;
        // cv::fisheye::undistortImage(img, img_undistorted, K_mat, D_mat);

        // std::stringstream ss;
        // ss << "image " << idx;
        // cv::imshow(ss.str(), img);
        // cv::waitKey(0);
        // cv::destroyWindow(ss.str());

        std::optional<Frame> frame;
        if (img_pt.AS_w_from_gnss_cam) {
            const Eigen::Isometry3d world_from_cam_reference =
                Eigen::Isometry3d((parser.S_from_AS().matrix() * scale_mat_reference *
                                   img_pt.AS_w_from_gnss_cam->matrix()));
            references_world_from_cam.push_back(world_from_cam_reference);
            frame.emplace(id, img_undistorted, K, gtsam::Pose3(world_from_cam_reference.matrix()));
        } else {
            frame.emplace(id, img_undistorted, K);
            std::clog << "Warning: no reference data at img_pt " << idx << std::endl;
        }

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> kpts_descs =
            frontend.get_keypoints_and_descriptors(img_undistorted);
        KeypointsCV kpts;
        for (const cv::KeyPoint &kpt : kpts_descs.first) {
            kpts.push_back(kpt.pt);
        }
        std::cout << "idx " << idx << " has " << kpts.size() << " kpts" << std::endl;
        frame->add_keypoints(kpts);
        frame->assign_descriptors(kpts_descs.second);

        frames.push_back(*frame);

        id++;
    }

    robot::geometry::viz_scene(references_world_from_cam, std::vector<Eigen::Vector3d>(), true,
                               true, "References");

    // populate feature_tracks and lmk_id_map
    // TODO: "smart" matching, using initial poses to filter which pairs make sense to compute
    // matches
    FeatureTracks feature_tracks;
    FrameLandmarkIdMap lmk_id_map;
    LandmarkId lmk_id = 0;
    bool exhaustive = false;
    if (exhaustive) {
        for (size_t i = 0; i < indices.size() - 1; i++) {
            for (size_t j = i + 1; i < indices.size(); j++) {
                std::vector<cv::DMatch> matches =
                    frontend.get_matches(frames[i].get_descriptors(), frames[j].get_descriptors());
                // DIAL TO MESS WITH
                frontend.enforce_bijective_buffer_matches(matches);
                for (const cv::DMatch match : matches) {
                    const KeypointCV kpt_cam0 = frames[i].get_keypoints()[match.queryIdx];
                    const KeypointCV kpt_cam1 = frames[j].get_keypoints()[match.trainIdx];

                    auto key = std::make_pair(frames[i].id_, kpt_cam0);
                    if (lmk_id_map.find(key) != lmk_id_map.end()) {
                        auto id = lmk_id_map.at(key);
                        feature_tracks.at(id).obs_.emplace_back(frames[i].id_, kpt_cam0);
                        feature_tracks.at(id).obs_.emplace_back(frames[j].id_, kpt_cam1);
                    } else {
                        FeatureTrack feature_track(i, kpt_cam0);
                        feature_track.obs_.emplace_back(frames[j].id_, kpt_cam1);
                        feature_tracks.emplace(lmk_id, feature_track);
                        lmk_id_map.emplace(std::make_pair(frames[i].id_, kpt_cam0), lmk_id);
                        lmk_id++;
                    }
                }
            }
        }
    } else {  // successive only
        for (size_t i = 0; i < indices.size() - 1; i++) {
            std::vector<cv::DMatch> matches =
                frontend.get_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
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
    }

    // triangulate all of the points
    std::unordered_map<LandmarkId, gtsam::Point3>
        lmk_triangulated_map;  // points are p_kpt_in_world
    std::vector<gtsam::Point3> triangulated_lmks;
    for (const std::pair<LandmarkId, FeatureTrack> lmk_feat : feature_tracks) {
        std::vector<gtsam::Pose3> world_from_cams;
        std::vector<gtsam::Point2> feat_kpts;
        const LandmarkId lmk_id = lmk_feat.first;
        const FeatureTrack feature_track = lmk_feat.second;
        gtsam::Point3 p_kpt_in_world(0, 0, 0);
        for (const std::pair<FrameId, KeypointCV> &feat_track : feature_track.obs_) {
            world_from_cams.push_back(id_to_initial_world_from_cam[feat_track.first]);
            feat_kpts.push_back(gtsam::Point2(feat_track.second.x, feat_track.second.y));
        }
        bool triangulate_success =
            detail_sfm::attempt_triangulate(world_from_cams, feat_kpts, K, p_kpt_in_world);
        if (triangulate_success) {
            lmk_triangulated_map.emplace(lmk_id, p_kpt_in_world);
            triangulated_lmks.push_back(p_kpt_in_world);
        } else {
            continue;
        }
    }
    // robot::geometry::viz_scene(std::vector<Eigen::Isometry3d>(), triangulated_lmks, true, true,
    //                            "Unfiltered, triangulated points");
    std::cout << "heart beat 1" << std::endl;

    // filter points via variance
    // TODO: filter based on quality and/or quantity of matches
    std::cout << "triangulated_lmks.size(): " << triangulated_lmks.size() << std::endl;
    const gtsam::Point3 variance_pts = detail_sfm::get_variance(triangulated_lmks);
    std::cout << "heart beat 2" << std::endl;
    const gtsam::Point3 std_dev_pts = variance_pts.array().sqrt().matrix();
    const gtsam::Point3 mean_pts = detail_sfm::averagePoints(triangulated_lmks);
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
    std::cout << "filtered variance " << detail_sfm::get_variance(filtered_points) << std::endl;
    // robot::geometry::viz_scene(std::vector<Eigen::Isometry3d>(), filtered_points, true, true,
    //                            "Unfiltered points");

    // ############# BACKEND ###############

    // add filtered points to graph
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Pose3>> symbols_poses_values_iter;
    std::unordered_map<gtsam::Symbol, std::vector<gtsam::Point3>> symbols_landmarks_values_iter;
    std::vector<gtsam::Symbol> symbols_pose;
    std::vector<gtsam::Symbol> symbols_landmarks;
    for (const auto &[lmk_id, p_kpt_in_world] : lmk_triangulated_map_filtered) {
        // LandmarkId lmk_id = lmk_id_pt.first;
        // const gtsam::Point3 p_kpt_in_world = lmk_id_pt.second;
        FeatureTrack feature_track = feature_tracks.at(lmk_id);
        const gtsam::Symbol symbol_lmk('l', lmk_id);
        for (const auto &[frame_id, obs] : feature_track.obs_) {
            initial_estimate_.insert_or_assign(symbol_lmk, p_kpt_in_world);
            symbols_landmarks_values_iter.emplace(symbol_lmk,
                                                  std::vector<gtsam::Point3>{p_kpt_in_world});
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

    // detail_sfm::graph_values(initial_estimate_, "Confirmation", symbols_pose,
    // symbols_landmarks);
    std::cout << "heart beat 3" << std::endl;

    bool local_optimizations = true;
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

            std::vector<cv::DMatch> matches =
                frontend.get_matches(frames[i].get_descriptors(), frames[i + 1].get_descriptors());
            // DIAL TO MESS WITH
            frontend.enforce_bijective_matches(matches);
            std::vector<gtsam::Pose3> world_from_cams{id_to_initial_world_from_cam.at(i),
                                                      id_to_initial_world_from_cam.at(i + 1)};

            std::vector<Eigen::Isometry3d> viz_world_from_cams{
                Eigen::Isometry3d(world_from_cams[0].matrix()),
                Eigen::Isometry3d(world_from_cams[1].matrix())};
            std::vector<Eigen::Vector3d> viz_lmks;
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

                gtsam::Point3 p_kpt_in_world;
                bool triangulate_success =
                    detail_sfm::attempt_triangulate(world_from_cams, feat_kpts, K, p_kpt_in_world);
                if (triangulate_success) {
                    local_estimate_.insert_or_assign(symbol_lmk, p_kpt_in_world);
                    viz_lmks.push_back(p_kpt_in_world);
                }
                if (symbols_landmarks_values_iter.find(symbol_lmk) !=
                    symbols_landmarks_values_iter.end()) {
                    symbols_landmarks_values_iter[symbol_lmk].push_back(p_kpt_in_world);
                } else {
                    symbols_landmarks_values_iter.emplace(
                        symbol_lmk, std::vector<gtsam::Point3>{p_kpt_in_world});
                }
            }
            std::cout << "setup complete!" << std::endl;
            // robot::geometry::viz_scene(viz_world_from_cams, viz_lmks, true, true,
            //                     "Local Optimization " + std::to_string(i));

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
        detail_sfm::optimize_graph(graph_, initial_estimate_, symbols_pose, symbols_landmarks);
    std::cout << "about to visualize result" << std::endl;
    detail_sfm::graph_values(result, "Result", symbols_pose, symbols_landmarks);
}