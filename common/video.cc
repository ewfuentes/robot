#include "opencv2/opencv.hpp"

namespace robot::common {

bool images_equal(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.reshape(1);
    return cv::countNonZero(diff) == 0;
}

}
