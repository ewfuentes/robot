#pragma once

#include <stdio.h>

#include <cstddef>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/image_point.hh"

namespace robot::experimental::learn_descriptors {
class FourSeasonsParser {
   public:
    struct CameraCalibrationFisheye {
        double fx, fy, cx, cy, k1, k2, k3, k4;
    };
    FourSeasonsParser(const std::filesystem::path& root_dir,
                      const std::filesystem::path& calibration_dir);
    cv::Mat load_image(const size_t idx) const;
    const ImagePoint& get_image_point(const size_t idx) const { return img_pt_vector_[idx]; };
    size_t num_images() const { return img_pt_vector_.size(); };
    const CameraCalibrationFisheye& get_camera_calibration() const { return cal_; };
    const liegroups::SE3& get_S_from_AS() const {
        return transforms_.S_from_AS;
    };  // metric scale from arbitrary (internal slam) scale
    const liegroups::SE3& get_cam_from_imu() const { return transforms_.cam_from_imu; };
    const liegroups::SE3& get_w_from_gpsw() const {
        return transforms_.w_from_gpsw;
    };  // visual world from local gps (ENU)
    const liegroups::SE3& get_gps_from_imu() const { return transforms_.gps_from_imu; };
    const liegroups::SE3& get_e_from_gpsw() const {
        return transforms_.e_from_gpsw;
    };  // ECEF from local gps (ENU)
    double get_gnss_scale() const {
        return transforms_.gnss_scale;
    };  // scale from vio frame to gnss frame. WARNING: will require retooling if the scales per
        // keyframe (pose) are not all one value. See more here:
        // https://github.com/pmwenzel/4seasons-dataset

    static Eigen::Vector3d gcs_from_ECEF(const Eigen::Vector3d& t_place_from_ECEF);
    static Eigen::Vector3d ECEF_from_gcs(const Eigen::Vector3d& gcs_coordinate);

   protected:
    struct FourSeasonsTransforms {
        liegroups::SE3 S_from_AS;
        liegroups::SE3 cam_from_imu;
        liegroups::SE3 w_from_gpsw;
        liegroups::SE3 gps_from_imu;
        liegroups::SE3 e_from_gpsw;
        double gnss_scale;

        FourSeasonsTransforms(const std::filesystem::path& path_transforms);

       private:
        static liegroups::SE3 get_transform_from_line(const std::string& line);
    };
    const std::filesystem::path root_dir_;
    const std::filesystem::path img_dir_;
    const CameraCalibrationFisheye cal_;
    const FourSeasonsTransforms transforms_;
    ImagePointVector img_pt_vector_;
};
}  // namespace robot::experimental::learn_descriptors
