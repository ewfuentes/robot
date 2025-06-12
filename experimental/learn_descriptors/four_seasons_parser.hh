#pragma once

#include <stdio.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "experimental/learn_descriptors/image_point.hh"

namespace robot::experimental::learn_descriptors {
class FourSeasonsParser {
   public:
    struct CameraCalibrationFisheye {
        double fx, fy, cx, cy, k1, k2, k3, k4;
    };
    FourSeasonsParser(const std::filesystem::path& root_dir,
                      const std::filesystem::path& calibration_dir);
    cv::Mat load_image(const size_t idx);
    const ImagePoint& get_image_point(const size_t idx);
    size_t num_images() { return img_pt_vector_.size(); };
    size_t size() { return num_images(); };
    const CameraCalibrationFisheye& get_camera_calibration() { return cal_; };

   protected:
    const std::filesystem::path root_dir_;
    const std::filesystem::path img_dir_;
    const CameraCalibrationFisheye cal_;
    ImagePointVector img_pt_vector_;
};
}  // namespace robot::experimental::learn_descriptors
