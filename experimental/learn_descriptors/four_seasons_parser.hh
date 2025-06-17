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
    const liegroups::SE3& get_AS_from_S() const { return transforms_.AS_from_S; };
    const liegroups::SE3& get_imu_from_cam() const { return transforms_.imu_from_cam; };
    const liegroups::SE3& get_gpsw_from_w() const { return transforms_.gpsw_from_w; };
    const liegroups::SE3& get_imu_from_gps() const { return transforms_.imu_from_gps; };
    const liegroups::SE3& get_gpsw_from_e() const { return transforms_.gpsw_from_e; };
    double get_gnss_scale() const { return transforms_.gnss_scale; };

   protected:
    struct FourSeasonsTransforms {
        liegroups::SE3 AS_from_S;
        liegroups::SE3 imu_from_cam;
        liegroups::SE3 gpsw_from_w;
        liegroups::SE3 imu_from_gps;
        liegroups::SE3 gpsw_from_e;
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
