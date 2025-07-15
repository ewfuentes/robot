#include "experimental/learn_descriptors/backend.hh"

#include <unordered_set>

#include "boost/make_shared.hpp"
#include "experimental/learn_descriptors/feature_manager.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "gtsam/geometry/triangulation.h"
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
    struct FrameVertex {
        Frame& frame;
        std::unordered_map<FrameId, gtsam::Rot3>
            neighbor_id_to_rot;  // rot is frame_from_neighbor_frame
    };
    std::unordered_map<FrameId, FrameVertex> frame_tree;
    // populate all frames and add children to neighbors
    for (Frame& frame : frames) {
        FrameVertex frame_vertex{frame};
        for (const auto [other_frame_id, frame_from_other_frame] : frame.frame_from_other_frames_) {
            frame_vertex.neighbor_id_to_rot.emplace(other_frame_id, frame_from_other_frame);
        }
        frame_tree.emplace(frame.id_, frame_vertex);
    }
    // add parents to neighbors
    for (const Frame& frame : frames) {
        for (const) }
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;
    // TODO: somehow get more informative/more robust noise values
    // maybe: monte carlo sampling + reprojection, use essential matrix to guide covariance strength
    gtsam::noiseModel::Diagonal::shared_ptr noise =
        gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // radians

    std::unordered_set<FrameId> frames_seen;
    for (size_t i = 0; i < frames.size(); i++) {
        const Frame& frame = frames[i];
        if (i == 0) {
            initial.insert(gtsam::Symbol(symbol_char_rotation, i),
                           frame.world_from_cam_initial_guess_
                               ? *frame.world_from_cam_initial_guess_
                               : gtsam::Rot3::Identity());
        }
        std::vector<gtsam::Rot3> candidates_world_from_frame;
        for (const auto& [other_frame_id, frame_from_other_frame] :
             frame.frame_from_other_frames_) {
            if (frames_seen.find(other_frame_id) == frames_seen.end()) {
                candidates_world_from_frame.push_back(frame_from_other_frame);
            }
        }
        initial.insert(gtsam::Symbol(symbol_char_rotation, frame.id_),
                       average_rotations(candidates_world_from_frame));
        frames_seen.insert(frame.id_);
    }

    // Assume we have relative rotations Rij between node i and j
    std::vector<std::pair<size_t, size_t>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    std::vector<gtsam::Rot3> relativeRotations = {
        Rot3::RzRyRx(0.25, 0.0, 0.0), Rot3::RzRyRx(0.25, 0.0, 0.0), Rot3::RzRyRx(0.3, 0.0, 0.0),
        Rot3::RzRyRx(0.2, 0.0, 0.0)};

    // Add BetweenFactors for relative rotations
    for (size_t k = 0; k < edges.size(); ++k) {
        size_t i = edges[k].first;
        size_t j = edges[k].second;
        graph.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(
            gtsam::Symbol(Backend::symbol_char_rotation, i),
            gtsam::Symbol(Backend::symbol_char_rotation, j), relativeRotations[k], noise);
    }

    // Fix the first rotation
    initial.insert(gtsam::Symbol(Backend::symbol_char_rotation, 0), gtsam::Rot3::Identity());

    // Initialize other rotations naively
    for (size_t i = 1; i <= 4; ++i) {
        initial.insert(gtsam::Symbol(Backend::symbol_char_rotation, i),
                       gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0));
    }

    // Optimize
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    gtsam::Values result = optimizer.optimize();

    // Output results
    for (size_t i = 0; i <= 4; ++i) {
        gtsam::Rot3 Ri = result.at<gtsam::Rot3>(gtsam::Symbol(Backend::symbol_char_rotation, i));
        std::cout << "Rotation " << i << ": " << Ri.rpy().transpose() << " (roll pitch yaw radians)"
                  << std::endl;
    }
}
}  // namespace robot::experimental::learn_descriptors