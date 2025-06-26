#pragma once
#include <optional>

#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Pose3.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class Frame {
   public:
    Frame(const FrameId& id, const cv::Mat& img_undistorted, const gtsam::Cal3_S2::shared_ptr& K,
          const gtsam::Pose3& T_wrld_grnd_truth)
        : id_(id), img_(img_undistorted), K_(K), groundtruth_(T_wrld_grnd_truth) {}
    Frame(const FrameId& id, const cv::Mat& img_undistorted, const gtsam::Cal3_S2::shared_ptr& K)
        : id_(id), img_(img_undistorted), K_(K) {}

    void add_keypoints(const KeypointsCV& kpts);
    void assign_descriptors(const cv::Mat& descriptors);

    const KeypointsCV get_keypoints() { return kpts_; };
    const cv::Mat get_descriptors() { return descriptors_; };

    const FrameId id_;
    const cv::Mat img_;
    gtsam::Cal3_S2::shared_ptr K_;
    KeypointsCV kpts_;
    cv::Mat descriptors_;
    std::optional<gtsam::Pose3> groundtruth_;
};
}  // namespace robot::experimental::learn_descriptors