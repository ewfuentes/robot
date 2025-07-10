#pragma once
#include <optional>

#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "gtsam/base/Matrix.h"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class Frame {
   public:
    void add_keypoints(const KeypointsCV& kpts);
    void assign_descriptors(const cv::Mat& descriptors);

    const KeypointsCV get_keypoints() { return kpts_; };
    const cv::Mat get_descriptors() { return descriptors_; };

    const FrameId id_;
    const cv::Mat undistorted_img_;
    gtsam::Cal3_S2::shared_ptr K_;
    KeypointsCV kpts_;
    cv::Mat descriptors_;
    std::optional<gtsam::Pose3> world_from_cam_groundtruth_;
    std::optional<gtsam::Point3> cam_in_world_initial_guess_;
    std::optional<gtsam::Matrix3> translation_covariance_in_cam_;
    std::optional<gtsam::Rot3> world_from_cam_initial_guess_;
};
}  // namespace robot::experimental::learn_descriptors