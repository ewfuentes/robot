#pragma once

#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
struct CameraCalibrationFisheye {
    CameraCalibrationFisheye(double fx, double fy, double cx, double cy, double k1, double k2,
                             double k3, double k4)
        : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), k3(k3), k4(k4) {}
    double fx, fy, cx, cy, k1, k2, k3, k4;

    cv::Mat k_mat() {
        cv::Mat K_mat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        return K_mat;
    }

    cv::Mat d_mat() {
        cv::Mat D_mat = (cv::Mat_<double>(4, 1) << k1, k2, k3, k4);
        return D_mat;
    }
};
}  // namespace robot::experimental::learn_descriptors