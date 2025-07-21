#pragma once
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "Eigen/Core"
#include "common/check.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "gtsam/base/Matrix.h"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class Frame {
   public:
    Frame(FrameId id, size_t seq, const cv::Mat img, gtsam::Cal3_S2::shared_ptr K,
          const KeypointsCV kpts, const cv::Mat descriptors)
        : id_(id),
          seq_(seq),
          undistorted_img_(img),
          K_(std::move(K)),
          kpts_(kpts),
          descriptors_(descriptors) {}

    std::optional<Eigen::Vector3d> velocity_to(const Frame& other_frame) const {
        ROBOT_CHECK(other_frame.seq_ >= seq_);
        if (!other_frame.cam_in_world_initial_guess_ || !cam_in_world_initial_guess_)
            return std::nullopt;
        double dt_seconds = static_cast<double>(other_frame.seq_ - seq_) * 1e-9;
        return (*other_frame.cam_in_world_initial_guess_ - *cam_in_world_initial_guess_) /
               dt_seconds;
    };
    std::optional<Eigen::Vector3d> velocity_from(const Frame& other_frame) const {
        ROBOT_CHECK(seq_ >= other_frame.seq_);
        if (!other_frame.cam_in_world_initial_guess_ || !cam_in_world_initial_guess_)
            return std::nullopt;
        double dt_seconds = static_cast<double>(seq_ - other_frame.seq_) * 1e-9;
        return (*cam_in_world_initial_guess_ - *other_frame.cam_in_world_initial_guess_) /
               dt_seconds;
    };

    void add_keypoints(const KeypointsCV& kpts);
    void assign_descriptors(const cv::Mat& descriptors);

    const KeypointsCV& keypoint() { return kpts_; };
    const cv::Mat& descriptors() { return descriptors_; };

    std::string to_string() const {
        std::stringstream ss;
        ss << "Frame " << id_ << ":\n";
        ss << "\tseq: " << seq_ << "\n";
        ss << "\tK: ";
        if (K_) {
            ss << "fx: " << K_->fx() << ", fy: " << K_->fy() << ", s: " << K_->skew()
               << ", px: " << K_->px() << ", py: " << K_->py();
        } else {
            ss << "N/A";
        }
        ss << "\n\tkpts_.size(): " << kpts_.size();
        ss << "\n\tframe_from_other_frames_.size(): " << frame_from_other_frames_.size();
        ss << "\n\tworld_from_cam_groundtruth_: ";
        if (world_from_cam_groundtruth_) {
            ss << "\n" << world_from_cam_groundtruth_->matrix();
        } else {
            ss << "N/A";
        }
        ss << "\n\tcam_in_world_initial_guess_: ";
        if (cam_in_world_initial_guess_) {
            ss << "\n" << cam_in_world_initial_guess_->matrix();
        } else {
            ss << "N/A";
        }
        ss << "\n\tworld_from_cam_initial_guess_: ";
        if (world_from_cam_initial_guess_) {
            ss << "\n" << world_from_cam_initial_guess_->matrix();
        } else {
            ss << "N/A";
        }
        ss << "\n\ttranslation_covariance_in_cam_: ";
        if (translation_covariance_in_cam_) {
            ss << "\n" << translation_covariance_in_cam_->matrix();
        } else {
            ss << "N/A";
        }
        return ss.str();
    }

    FrameId id_;
    size_t seq_;
    cv::Mat undistorted_img_;
    gtsam::Cal3_S2::shared_ptr K_;
    KeypointsCV kpts_;
    cv::Mat descriptors_;
    std::unordered_map<FrameId, gtsam::Rot3>
        frame_from_other_frames_;  // map of relative rotation for this_frame_from_frame_[FrameId]
    std::optional<gtsam::Pose3> world_from_cam_groundtruth_;
    std::optional<gtsam::Point3> cam_in_world_initial_guess_;
    std::optional<gtsam::Rot3> world_from_cam_initial_guess_;
    std::optional<gtsam::Matrix3> translation_covariance_in_cam_;
};
}  // namespace robot::experimental::learn_descriptors