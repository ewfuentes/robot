#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/camera_calibration.hh"
#include "experimental/learn_descriptors/four_seasons_transforms.hh"
#include "experimental/learn_descriptors/image_point_four_seasons.hh"

namespace robot::experimental::learn_descriptors {
class FourSeasonsParser : std::enable_shared_from_this<FourSeasonsParser> {
   public:
    static constexpr double CAM_HZ = 30.0;
    static constexpr double CAM_CAP_DELTA_NS = 1e9 / CAM_HZ;

    FourSeasonsParser(const std::filesystem::path& root_dir,
                      const std::filesystem::path& calibration_dir);
    std::shared_ptr<FourSeasonsParser> get_shared() { return shared_from_this(); }
    cv::Mat load_image(const size_t idx) const;
    const ImagePointFourSeasons& get_image_point(const size_t idx) const {
        return img_pt_vector_[idx];
    };
    size_t num_images() const { return img_pt_vector_.size(); };
    const std::shared_ptr<CameraCalibrationFisheye>& camera_calibration() const { return cal_; };
    const liegroups::SE3& S_from_AS() const {
        return shared_transforms_->S_from_AS;
    };  // metric scale from arbitrary (internal slam) scale
    const liegroups::SE3& cam_from_imu() const { return shared_transforms_->cam_from_imu; };
    const liegroups::SE3& w_from_gpsw() const {
        return shared_transforms_->w_from_gpsw;
    };  // visual world from local gps (ENU)
    const liegroups::SE3& gps_from_imu() const {
        return shared_transforms_->gps_from_imu;
    };  // phsyical onboard gps from physical onboard imu
    const liegroups::SE3& e_from_gpsw() const {
        return shared_transforms_->e_from_gpsw;
    };  // ECEF from local gps (ENU)
    double gnss_scale() const {
        return shared_transforms_->gnss_scale;
    };  // scale from vio frame to gnss frame. WARNING: will require retooling if the scales per
        // keyframe (pose) are not all one value. See more here:
        // https://github.com/pmwenzel/4seasons-dataset

   protected:
    const std::filesystem::path root_dir_;
    const std::filesystem::path img_dir_;
    const std::shared_ptr<CameraCalibrationFisheye> cal_;
    const std::shared_ptr<FourSeasonsTransforms::StaticTransforms> shared_transforms_;
    ImagePointFourSeasonsVector img_pt_vector_;
};
}  // namespace robot::experimental::learn_descriptors
