#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Point2.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"

namespace robot::experimental::learn_descriptors {
class Backend {
   public:
    static constexpr char symbol_char_pose = 'x';
    static constexpr char symbol_char_rotation = 'r';
    static constexpr char symbol_char_translation = 't';
    static constexpr char symbol_char_bearing = 'b';
    static constexpr char symbol_char_landmark = 'l';
    static constexpr char symbol_char_cam_cal = 'k';

    static std::optional<gtsam::Point3> attempt_triangulate(
        const std::vector<gtsam::Pose3> &cam_poses, const std::vector<gtsam::Point2> &cam_obs,
        gtsam::Cal3_S2::shared_ptr K, const double max_reproj_error = 2.0,
        const bool verbose = true);
    static gtsam::Rot3 average_rotations(const std::vector<gtsam::Rot3> &rotations,
                                         int max_iter = 10);
    // use rotation averaging to set the rotation initial guess for each frame in the world frame.
    // the world frame will be the first frame in the vector. deadreckon_incrementally won't
    // optimize if true
    static void populate_rotation_estimate(std::vector<Frame> &frames);
    // use rotation averaging to set the rotation initial guess for each frame in the world frame.
    // the world frame will be the first frame in the vector. deadreckon_incrementally won't
    // optimize if true
    static void populate_rotation_estimate(std::vector<SharedFrame> &shared_frames);

    void add_frames(std::vector<SharedFrame> &shared_frames) {
        shared_frames_.reserve(shared_frames_.size() + shared_frames.size());
        shared_frames_.insert(shared_frames_.end(), shared_frames.begin(), shared_frames.end());
    };
    void calculate_initial_values(bool interpolate_gps = true);
    void populate_graph(const FeatureTracks &feature_tracks);
    typedef int epoch;
    using graph_step_debug_func = std::function<void(const gtsam::Values &, const epoch)>;
    void solve_graph(const int num_epochs,
                     std::optional<graph_step_debug_func> iter_debug_func = std::nullopt);
    void clear();

    const std::vector<SharedFrame> &shared_frames() const { return shared_frames_; };
    const gtsam::Values &current_initial_values() const { return initial_estimate_; };
    const gtsam::Values &result() const { return result_; };

   private:
    std::vector<SharedFrame> shared_frames_;

    gtsam::Values initial_estimate_;
    gtsam::Values result_;
    gtsam::NonlinearFactorGraph graph_;

    std::unordered_map<size_t, gtsam::Pose3> world_from_cam_initial_estimates_;
    std::unordered_map<size_t, gtsam::Point3> lmk_initial_estimates_;

    gtsam::noiseModel::Diagonal::shared_ptr noise_tight_prior_ =
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3,
                                             1e-3,             // rotation stdev in radians
                                             1e-3, 1e-3, 1e-3  // translation stdev in meters
                                             )
                                                .finished());
    gtsam::Vector3 gps_sigmas_fallback_{0.5, 0.5, 0.5};
    gtsam::noiseModel::Isotropic::shared_ptr landmark_noise_ =
        gtsam::noiseModel::Isotropic::Sigma(2, 10.0);
};
using Landmark = gtsam::Point3;
}  // namespace robot::experimental::learn_descriptors