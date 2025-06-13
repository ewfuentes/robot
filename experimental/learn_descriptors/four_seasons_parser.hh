#pragma once

#include <stdio.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
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
    size_t size() const { return num_images(); };
    const CameraCalibrationFisheye& get_camera_calibration() const { return cal_; };
    const Eigen::Isometry3d& get_T_S_AS() const { return transforms_.T_S_AS; };
    const Eigen::Isometry3d& get_T_cam_imu() const { return transforms_.T_cam_imu; };
    const Eigen::Isometry3d& get_T_gps_imu() const { return transforms_.T_gps_imu; };
    const Eigen::Isometry3d& get_T_e_gpsw() const { return transforms_.T_e_gpsw; };
    double get_gnss_scale() const { return transforms_.gnss_scale; };

   protected:
    struct FourSeasonsTransforms {
        Eigen::Isometry3d T_S_AS;
        Eigen::Isometry3d T_cam_imu;
        Eigen::Isometry3d T_w_gpsw;
        Eigen::Isometry3d T_gps_imu;
        Eigen::Isometry3d T_e_gpsw;
        double gnss_scale;

        FourSeasonsTransforms(const std::filesystem::path& path_transforms);

       private:
        static Eigen::Isometry3d get_transform_from_line(const std::string& line);
    };
    const std::filesystem::path root_dir_;
    const std::filesystem::path img_dir_;
    const CameraCalibrationFisheye cal_;
    const FourSeasonsTransforms transforms_;
    ImagePointVector img_pt_vector_;
};
}  // namespace robot::experimental::learn_descriptors
