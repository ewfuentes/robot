#pragma once

#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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
    const liegroups::SE3& get_T_S_AS() const { return transforms_.T_S_AS; };
    const liegroups::SE3& get_T_cam_imu() const { return transforms_.T_cam_imu; };
    const liegroups::SE3& get_T_w_gpsw() const { return transforms_.T_w_gpsw; };
    const liegroups::SE3& get_T_gps_imu() const { return transforms_.T_gps_imu; };
    const liegroups::SE3& get_T_e_gpsw() const { return transforms_.T_e_gpsw; };
    double get_gnss_scale() const { return transforms_.gnss_scale; };

   protected:
    struct FourSeasonsTransforms {
        liegroups::SE3 T_S_AS;
        liegroups::SE3 T_cam_imu;
        liegroups::SE3 T_w_gpsw;
        liegroups::SE3 T_gps_imu;
        liegroups::SE3 T_e_gpsw;
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
