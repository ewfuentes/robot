#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "Eigen/Dense"
#include "common/geometry/camera.hh"
#include "experimental/learn_descriptors/backend.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "experimental/learn_descriptors/image_point.hh"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/Values.h"
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"

namespace robot::experimental::learn_descriptors {
class StructureFromMotion {
   public:
    StructureFromMotion(FrontendParams frontend_params) : frontend_(Frontend(frontend_params)){};

    static void graph_values(const gtsam::Values &values,
                             const std::string &window_name = "Viz Structure",
                             const cv::viz::Color color = cv::viz::Color::brown());

    void add_image_point(const cv::Mat &img_undistorted, const SharedImagePoint &image_point);
    void add_image_points(const std::vector<cv::Mat> &imgs_undistorted,
                          const std::vector<SharedImagePoint> &image_points);
    void solve_structure(
        const int num_steps,
        std::optional<Backend::graph_step_debug_func> iter_debug_func = std::nullopt);
    const gtsam::Values &result() { return backend_.result(); };

    Frontend frontend() { return frontend_; };
    Backend backend() { return backend_; }

   private:
    Frontend frontend_;
    Backend backend_;
};
}  // namespace robot::experimental::learn_descriptors