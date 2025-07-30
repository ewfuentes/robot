#include "experimental/learn_descriptors/backend.hh"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_set>

#include "boost/make_shared.hpp"
#include "common/check.hh"
#include "experimental/learn_descriptors/frame.hh"
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

namespace robot::experimental::learn_descriptors {

std::optional<gtsam::Point3> Backend::attempt_triangulate(
    const std::vector<gtsam::Pose3>& cam_poses, const std::vector<gtsam::Point2>& cam_obs,
    gtsam::Cal3_S2::shared_ptr K, const double max_reproj_error, const bool verbose) {
    gtsam::Point3 p_lmk_in_world;
    if (cam_poses.size() > 2) {
        try {
            // Attempt triangulation using DLT (or the GTSAM provided method)
            p_lmk_in_world = gtsam::triangulatePoint3(
                cam_poses, K, gtsam::Point2Vector(cam_obs.begin(), cam_obs.end()));

        } catch (const std::exception& e) {
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
    for (const auto& pose : cam_poses) {
        // Transform point to the camera coordinate system.
        // transformTo() converts a world point to the camera frame.
        gtsam::Point3 p_cam_lmk = pose.transformTo(p_lmk_in_world);
        if (p_cam_lmk.z() <= 0) {  // Check that the depth is positive
            return std::nullopt;
        }
    }

    // Cheirality & reprojection checks
    for (size_t i = 0; i < cam_poses.size(); ++i) {
        const auto& pose = cam_poses[i];
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

gtsam::Rot3 average_rotations(const std::vector<gtsam::Rot3>& rotations, int max_iter = 10) {
    if (rotations.empty()) throw std::runtime_error("No rotations to average");
    if (rotations.size() == 1) return rotations.front();

    gtsam::Rot3 mean = rotations[0];

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::Vector3d total = Eigen::Vector3d::Zero();
        for (const gtsam::Rot3& R : rotations) {
            gtsam::Rot3 delta = mean.between(R);  // R_mean^T * R
            total += gtsam::Rot3::Logmap(delta);  // Lie algebra element in R³
        }
        total /= static_cast<double>(rotations.size());
        mean = mean.compose(gtsam::Rot3::Expmap(total));  // update estimate
    }

    return mean;
}

void Backend::populate_rotation_estimate(std::vector<SharedFrame>& shared_frames) {
    struct FrameVertex;
    using SharedFrameVertex = std::shared_ptr<FrameVertex>;
    struct FrameVertex {
        Frame& frame;
        std::optional<gtsam::Rot3> w_from_this;
        std::unordered_map<FrameId, gtsam::Rot3>
            neighbor_id_to_rot;  // rot is frame_from_neighbor_frame
        std::unordered_set<SharedFrameVertex> parents;
        std::unordered_set<SharedFrameVertex> children;
    };
    std::unordered_map<FrameId, SharedFrameVertex> frame_tree;
    for (SharedFrame& shared_frame : shared_frames) {
        frame_tree.emplace(shared_frame->id_, std::make_shared<FrameVertex>(*shared_frame));
    }
    // populate all frames and add children to neighbors
    std::queue<size_t> q;  // indices of Frame in frames
    std::unordered_set<FrameId> seen_frame_id;
    q.push(shared_frames.front()->id_);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    // TODO: somehow get more informative/more robust noise values
    // maybe: monte carlo sampling + reprojection, use essential matrix to guide covariance strength
    gtsam::noiseModel::Diagonal::shared_ptr noise_rotation =
        gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // radians

    std::vector<gtsam::Symbol> symbols_frames;
    while (!q.empty()) {
        Frame frame = *shared_frames[q.front()];
        q.pop();
        seen_frame_id.insert(frame.id_);
        SharedFrameVertex frame_vertex = frame_tree.at(frame.id_);
        if (seen_frame_id.size() == 1) {  // make the first frame rotation identity unless the
                                          // initial guess is set otherwise
            frame_vertex->w_from_this = frame_vertex->frame.world_from_cam_initial_guess_
                                            ? *frame_vertex->frame.world_from_cam_initial_guess_
                                            : gtsam::Rot3::Identity();
            graph.add(gtsam::PriorFactor<gtsam::Rot3>(
                gtsam::Symbol(symbol_char_rotation, frame_vertex->frame.id_),
                *frame_vertex->w_from_this, gtsam::noiseModel::Constrained::All(3)));
        }
        frame_vertex->frame.world_from_cam_initial_guess_ = frame_vertex->w_from_this;
        std::cout << frame.to_string() << std::endl;

        const gtsam::Symbol symbol_frame(symbol_char_rotation, frame.id_);
        symbols_frames.push_back(symbol_frame);
        initial.insert(symbol_frame, *frame.world_from_cam_initial_guess_);
        for (const auto& [other_frame_id, frame_from_other_frame] :
             frame.frame_from_other_frames_) {
            if (seen_frame_id.find(other_frame_id) == seen_frame_id.end()) {
                q.push(other_frame_id);
                seen_frame_id.insert(other_frame_id);
            }
            frame_vertex->neighbor_id_to_rot.emplace(other_frame_id, frame_from_other_frame);
            SharedFrameVertex other_frame_vertex = frame_tree.at(other_frame_id);
            frame_vertex->children.insert(other_frame_vertex);
            other_frame_vertex->parents.insert(frame_vertex);
            other_frame_vertex->neighbor_id_to_rot.emplace(frame.id_,
                                                           frame_from_other_frame.inverse());
            if (!other_frame_vertex->w_from_this) {
                other_frame_vertex->w_from_this =
                    *frame_vertex->w_from_this * frame_from_other_frame;
                other_frame_vertex->frame.world_from_cam_initial_guess_ =
                    other_frame_vertex->w_from_this;
            }

            graph.add(gtsam::BetweenFactor<gtsam::Rot3>(
                symbol_frame, gtsam::Symbol(symbol_char_rotation, other_frame_id),
                frame_from_other_frame.inverse(), noise_rotation));
        }
        frame_tree.emplace(frame.id_, frame_vertex);
    }

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    gtsam::Values result = optimizer.optimize();

    // Output results and assign results to frames
    for (const gtsam::Symbol& symbol_rotation : symbols_frames) {
        gtsam::Rot3 w_from_frame = result.at<gtsam::Rot3>(symbol_rotation);
        frame_tree.at(symbol_rotation.index())->frame.world_from_cam_initial_guess_ = w_from_frame;
    }
}

void Backend::calculate_initial_values(bool interpolate_gps) {
    populate_rotation_estimate(shared_frames_);
    if (interpolate_gps) {
        interpolate_gps();
    }
}

void Backend::interpolate_gps() {
    std::vector<SharedFrame> frames_with_gps;
    for (SharedFrame& shared_frame : shared_frames_) {  // this assumes that frames are ordered
                                                        // by seq (time), later on may have to
        // think about this more if trying to run pipeline on unordered images or sets
        // whose capture times are on different days
        if (shared_frame->cam_in_world_initial_guess_) {
            frames_with_gps.push_back(shared_frame);
        }
    }
    if (frames_with_gps.size() < 2) return;
    size_t idx_gps = 0;
    for (SharedFrame& shared_frame : shared_frames_) {
        // advance to next guess window if needed
        while (idx_gps + 1 < frames_with_gps.size() - 1 &&
               shared_frame->seq_ > frames_with_gps[idx_gps + 1]->seq_) {
            idx_gps++;
        }

        if (shared_frame->cam_in_world_initial_guess_) {
            continue;
        }
        ROBOT_CHECK(shared_frame->cam_in_world_interpolated_guess_,
                    "Currently assuming that the frontend populated interpolated initial guesses");
        Eigen::Matrix3d interpolated_covariance(
            *shared_frame->translation_covariance_in_cam_ *
            100.0);  // very jank for now, not true interpolation
        shared_frame->translation_covariance_in_cam_ = interpolated_covariance;
        shared_frame->cam_in_world_initial_guess_ = shared_frame->cam_in_world_interpolated_guess_;
    }
}

void Backend::graph_add_frame(const size_t idx_frame) {
    ROBOT_CHECK(idx_frame < shared_frames_.size());
    const Frame& frame = *shared_frames_[idx_frame];
    gtsam::Pose3 world_from_cam_estimate(frame.world_from_cam_initial_guess_
                                             ? *frame.world_from_cam_initial_guess_
                                             : gtsam::Rot3::Identity(),
                                         *front.cam_in_world_interpolated_guess_ - cam0_in_w_);
    world_from_cam_initial_estimates_.emplace(frame.id_, world_from_cam_estimate);

    const gtsam::Symbol cam_symbol(symbol_char_pose, frame.id_);
    values_.insert(cam_symbol, world_from_cam_initial_estimates_[frame.id_]);
    if (frame.cam_in_world_initial_guess_) {
        gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
            frame.translation_covariance_in_cam_
                ? gtsam::Vector3(std::sqrt((*frame.translation_covariance_in_cam_)(0, 0)),
                                 std::sqrt((*frame.translation_covariance_in_cam_)(1, 1)),
                                 std::sqrt((*frame.translation_covariance_in_cam_)(2, 2)))
                : gps_sigmas_fallback_);
        graph_.add(gtsam::GPSFactor(cam_symbol, *frame.cam_in_world_initial_guess_, gps_noise));
    }
}

void Backend::seed_graph() {
    ROBOT_CHECK(shared_frames_.size() > 3);
    ROBOT_CHECK(shared_frames_.front()->world_from_cam_groundtruth_,
                "We're assuming the first camera has groundtruth for now");
    shared_frames_.front()->world_from_cam_initial_guess_ =
        shared_frames_.front()
            ->world_from_cam_groundtruth_->rotation();  // align rotation with groundtruth

    // add first cam as prior
    cam0_in_w_ = *shared_frames_.front()->cam_in_world_interpolated_guess_;
    const gtsam::Pose3 w_from_cam0_init_estimate(
        *shared_frames_.front()->world_from_cam_initial_guess_, gtsam::Point3{0, 0, 0});
    const gtsam::Symbol symbol_cam0(symbol_char_pose, 0);
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
        symbol_cam0, w_from_cam0_init_estimate,
        noise_tight_prior_));  // currently assuming that we have groundtruth on the first pose
    values_.insert(symbol_cam0, w_from_cam0_init_estimate);

    // add second cam
    graph_add_frame(1);

    // add the matches
}

bool Backend::register_next_best_image() {
    if (shared_frames_.size() == frames_added_.size()) return false;
    graph_add_frame(frames_added_.size());
}

void Backend::bundle_adjust() {}

void Backend::populate_graph(const FeatureTracks& feature_tracks) {
    ROBOT_CHECK(shared_frames_.size() > 3);
    ROBOT_CHECK(shared_frames_.front()->world_from_cam_groundtruth_,
                "We're assuming the first camera has groundtruth for now");

    shared_frames_.front()->world_from_cam_initial_guess_ =
        shared_frames_.front()
            ->world_from_cam_groundtruth_->rotation();  // align rotation with groundtruth

    Backend::populate_rotation_estimate(shared_frames_);

    // add first cam as prior
    Eigen::Vector3d cam0_in_w = *shared_frames_.front()->cam_in_world_interpolated_guess_;
    const gtsam::Pose3 w_from_cam0_init_estimate(
        *shared_frames_.front()->world_from_cam_initial_guess_, gtsam::Point3{0, 0, 0});
    const gtsam::Symbol symbol_cam0(symbol_char_pose, 0);
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
        symbol_cam0, w_from_cam0_init_estimate,
        noise_tight_prior_));  // currently assuming that we have groundtruth on the first pose
    values_.insert(symbol_cam0, w_from_cam0_init_estimate);

    // add gps factors and populate initial pose values
    for (size_t i = 1; i < shared_frames_.size(); i++) {
        const Frame& frame = *shared_frames_[i];
        gtsam::Pose3 world_from_cam_estimate(frame.world_from_cam_initial_guess_
                                                 ? *frame.world_from_cam_initial_guess_
                                                 : gtsam::Rot3::Identity(),
                                             *frame.cam_in_world_interpolated_guess_ - cam0_in_w);
        world_from_cam_initial_estimates_.emplace(frame.id_, world_from_cam_estimate);

        const gtsam::Symbol cam_symbol(symbol_char_pose, frame.id_);
        values_.insert(cam_symbol, world_from_cam_initial_estimates_[frame.id_]);
        if (frame.cam_in_world_initial_guess_) {
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Sigmas(
                frame.translation_covariance_in_cam_
                    ? gtsam::Vector3(std::sqrt((*frame.translation_covariance_in_cam_)(0, 0)),
                                     std::sqrt((*frame.translation_covariance_in_cam_)(1, 1)),
                                     std::sqrt((*frame.translation_covariance_in_cam_)(2, 2)))
                    : gps_sigmas_fallback_);
            graph_.add(gtsam::GPSFactor(cam_symbol, *frame.cam_in_world_initial_guess_, gps_noise));
        }
    }

    // add landmarks
    for (size_t i = 0; i < feature_tracks.size(); i++) {
        std::vector<gtsam::Pose3> world_from_lmk_cams;
        std::vector<gtsam::Point2> lmk_observations;
        for (const auto& [frame_id, keypoint_cv] : feature_tracks[i].obs_) {
            world_from_lmk_cams.push_back(world_from_cam_initial_estimates_[frame_id]);
            lmk_observations.emplace_back(keypoint_cv.x, keypoint_cv.y);
        }
        std::optional<gtsam::Point3> landmark_estimate =
            attempt_triangulate(world_from_lmk_cams, lmk_observations, shared_frames_[0]->K_, 2.0,
                                false);  // all K are the same for now...
        if (landmark_estimate) {
            const gtsam::Symbol lmk_symbol(symbol_char_landmark, i);
            for (const auto& [frame_id, keypoint_cv] : feature_tracks[i].obs_) {
                graph_.add(gtsam::GenericProjectionFactor(
                    gtsam::Point2(keypoint_cv.x, keypoint_cv.y), landmark_noise_,
                    gtsam::Symbol(symbol_char_pose, frame_id), lmk_symbol,
                    shared_frames_[0]->K_));  // all K are the same for now...
            }
            values_.insert(lmk_symbol, *landmark_estimate);
            lmk_initial_estimates_.emplace(i, *landmark_estimate);
        }
    }
}

void Backend::solve_graph(const int num_epochs,
                          std::optional<Backend::graph_step_debug_func> iter_debug_func) {
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
    params.maxIterations = 1;          // We'll manually step it
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, params);

    double prev_error = optimizer.error();
    double lowest_error = prev_error;
    gtsam::Values best_values = optimizer.values();
    for (int i = 0; i < num_epochs; i++) {
        optimizer.iterate();
        double curr_error = optimizer.error();
        if (curr_error < lowest_error) {
            lowest_error = curr_error;
            best_values = optimizer.values();
        }

        if (iter_debug_func) {
            (*iter_debug_func)(optimizer.values(), i);
        }
        // if (std::abs(prev_error - curr_error) < 1e-6) {
        //     std::cout << "Converged at iteration " << i << "\n";
        //     break;
        // }
    }
    result_ = best_values;
}

void Backend::clear() {
    shared_frames_.clear();
    shared_frames_.shrink_to_fit();

    values_ = gtsam::Values();
    result_ = gtsam::Values();
    graph_ = gtsam::NonlinearFactorGraph();

    std::unordered_map<size_t, gtsam::Pose3>().swap(world_from_cam_initial_estimates_);
    std::unordered_map<size_t, gtsam::Point3>().swap(lmk_initial_estimates_);
}
}  // namespace robot::experimental::learn_descriptors