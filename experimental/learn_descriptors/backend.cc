#include "experimental/learn_descriptors/backend.hh"

#include <memory>
#include <optional>
#include <queue>
#include <unordered_set>

#include "boost/make_shared.hpp"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "gtsam/geometry/triangulation.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/GPSFactor.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"

namespace robot::experimental::learn_descriptors {

void Backend::solve_graph() {
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_);
    result_ = optimizer.optimize();
}

void Backend::solve_graph(const int num_steps,
                          std::optional<Backend::graph_step_debug_func> inter_debug_func) {
    gtsam::LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");  // or "TERMINATION", "TRYLAMBDA", etc.
    params.maxIterations = 1;          // We'll manually step it
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_, params);

    double prev_error = optimizer.error();
    for (int i = 0; i < num_steps; i++) {
        optimizer.iterate();
        double curr_error = optimizer.error();

        if (inter_debug_func) {
            (*inter_debug_func)(optimizer.values(), i);
        }
        if (std::abs(prev_error - curr_error) < 1e-6) {
            std::cout << "Converged at iteration " << i << "\n";
            break;
        }
    }
    result_ = optimizer.values();
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

void Backend::populate_rotation_estimate(std::vector<Frame>& frames) {
    struct FrameVertex;
    using SharedFrameVertex = std::shared_ptr<FrameVertex>;
    struct FrameVertex {
        // std::unique_ptr<Frame> frame;
        Frame& frame;
        std::optional<gtsam::Rot3> w_from_this;
        std::unordered_map<FrameId, gtsam::Rot3>
            neighbor_id_to_rot;  // rot is frame_from_neighbor_frame
        std::unordered_set<SharedFrameVertex> parents;
        std::unordered_set<SharedFrameVertex> children;
    };
    std::unordered_map<FrameId, SharedFrameVertex> frame_tree;
    for (Frame& frame : frames) {
        frame_tree.emplace(frame.id_, std::make_shared<FrameVertex>(frame));
    }
    // populate all frames and add children to neighbors
    std::queue<size_t> q;  // indices of Frame in frames
    std::unordered_set<FrameId> seen_frame_id;
    q.push(frames.front().id_);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    // TODO: somehow get more informative/more robust noise values
    // maybe: monte carlo sampling + reprojection, use essential matrix to guide covariance strength
    gtsam::noiseModel::Diagonal::shared_ptr noise_rotation =
        gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // radians

    std::vector<gtsam::Symbol> symbols_frames;
    while (!q.empty()) {
        Frame frame = frames[q.front()];
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

    // gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    // gtsam::Values result = optimizer.optimize();

    // // Output results and assign results to frames
    // for (const gtsam::Symbol& symbol_rotation : symbols_frames) {
    //     gtsam::Rot3 w_from_frame = result.at<gtsam::Rot3>(symbol_rotation);
    //     frame_tree.at(symbol_rotation.index())->frame.world_from_cam_initial_guess_ =
    //     w_from_frame;
    //     // std::cout << "Rotation " << symbol_rotation.string() << ": "
    //     //           << w_from_frame.rpy().transpose() << " (roll pitch yaw radians)" <<
    //     std::endl;
    // }
}
}  // namespace robot::experimental::learn_descriptors