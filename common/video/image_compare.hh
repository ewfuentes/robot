#pragma once

#include "opencv2/opencv.hpp"

namespace robot::common::video {

bool images_equal(const cv::Mat& img1, const cv::Mat& img2);

}
