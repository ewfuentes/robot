#pragma once

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace robot::experimental::learn_descriptors {
struct ImagePoint {
    size_t id;              // idx for DB
    size_t seq;             // time in nanoseconds (also the name of image)
    Eigen::Isometry3d gps;  // UTM. Not sure why the dataset has transforms for GPS, not sure if
                            // rotation or trans_z is reliable
    Eigen::Isometry3d ground_truth;

    bool has_gps = false;
    bool has_ground_truth = false;

    cv::Mat load_image(const std::filesystem::path &img_dir_) const {
        const std::filesystem::path path(img_dir_ / (std::to_string(seq) + ".png"));
        if (std::getenv("DEBUG")) {
            std::cout << "getting image at " << path << std::endl;
        }
        return cv::imread(path);
    }
};
typedef std::vector<ImagePoint> ImagePointVector;
}  // namespace robot::experimental::learn_descriptors