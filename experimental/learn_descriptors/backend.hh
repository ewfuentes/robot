#pragma once

#include <memory>
#include <optional>
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

    void solve_graph();
    typedef int epoch;
    using graph_step_debug_func = std::function<void(const gtsam::Values &, const epoch)>;
    void solve_graph(const int num_steps,
                     std::optional<graph_step_debug_func> inter_debug_func = std::nullopt);

    static gtsam::Rot3 average_rotations(const std::vector<gtsam::Rot3> &rotations,
                                         int max_iter = 10);
    // use rotation averaging to set the rotation initial guess for each frame in the world frame.
    // the world frame will be the first frame in the vector. deadreckon_incrementally won't
    // optimize if true
    static void populate_rotation_estimate(std::vector<Frame> &frames);

    const gtsam::Values &current_initial_values() const { return initial_estimate_; };
    const gtsam::Values &result() const { return result_; };

   private:
    gtsam::Values initial_estimate_;
    gtsam::Values result_;
    gtsam::NonlinearFactorGraph graph_;
};
using Landmark = gtsam::Point3;
}  // namespace robot::experimental::learn_descriptors