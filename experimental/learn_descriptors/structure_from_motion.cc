#include "experimental/learn_descriptors/structure_from_motion.hh"

#include <filesystem>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "common/check.hh"
#include "common/geometry/camera.hh"
#include "common/geometry/translate_types.hh"
#include "experimental/learn_descriptors/backend.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "gtsam/geometry/Point2.h"
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"
#include "visualization/opencv/opencv_viz.hh"

namespace robot::experimental::learn_descriptors {
void StructureFromMotion::add_image_point(const cv::Mat &img_undistorted,
                                          const SharedImagePoint &image_point) {
    frontend_.add_image(ImageAndPoint{img_undistorted, image_point});
}
void StructureFromMotion::add_image_points(
    const std::vector<cv::Mat> &imgs_undistorted,
    const std::vector<SharedImagePoint> &shared_image_points) {
    ROBOT_CHECK(imgs_undistorted.size() == shared_image_points.size());
    std::vector<ImageAndPoint> img_and_pts;
    img_and_pts.reserve(shared_image_points.size());
    for (size_t i = 0; i < shared_image_points.size(); i++) {
        img_and_pts.emplace_back(imgs_undistorted[i], shared_image_points[i]);
    }
    frontend_.add_images(img_and_pts);
}

void StructureFromMotion::solve_structure(
    const int num_steps, std::optional<Backend::graph_step_debug_func> iter_debug_func) {
    // FRONTEND
    frontend_.populate_frames();
    ROBOT_CHECK(frontend_.frames().size() > 2);
    frontend_.match_frames_and_build_tracks();
    ROBOT_CHECK(frontend_.frames().front()->world_from_cam_groundtruth_,
                "Assuming that our first frame has groundtruth for now.",
                *frontend_.frames().front()->world_from_cam_groundtruth_);
    // align rotation via the initial guess
    frontend_.frames().front()->world_from_cam_initial_guess_ =
        frontend_.frames().front()->world_from_cam_groundtruth_->rotation();

    // BACKEND
    backend_.clear();
    backend_.add_frames(frontend_.frames());
    backend_.calculate_initial_values();
    backend_.populate_graph(frontend_.feature_tracks());
    graph_values(backend_.current_initial_values(), "viz init values");
    backend_.solve_graph(10, iter_debug_func);
}

void StructureFromMotion::graph_values(const gtsam::Values &values, const std::string &window_name,
                                       const cv::viz::Color color) {
    std::vector<robot::visualization::VizPose> final_poses;
    std::vector<robot::visualization::VizPoint> final_lmks;
    for (const auto &key_value : values) {
        gtsam::Key key = key_value.key;
        gtsam::Symbol symbol(key);
        try {
            const gtsam::Pose3 &pose = values.at<gtsam::Pose3>(key);
            final_poses.emplace_back(Eigen::Isometry3d(pose.matrix()), symbol.string());
            continue;
        } catch (const gtsam::ValuesIncorrectType &) {
        }

        try {
            const gtsam::Point3 &pt = values.at<gtsam::Point3>(key);
            final_lmks.emplace_back(pt, symbol.string());
            continue;
        } catch (const gtsam::ValuesIncorrectType &) {
        }

        std::cerr << "Key " << symbol << " is neither Pose3 nor Point3.\n";
    }
    robot::visualization::viz_scene(final_poses, final_lmks, color, true, true, window_name);
}
}  // namespace robot::experimental::learn_descriptors