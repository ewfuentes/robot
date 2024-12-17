#include "common/video/image_compare.hh"

namespace robot::common::video {

bool images_equal(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    if (img1.type() == CV_32F) {
        cv::Mat img1_no_nan = img1;
        cv::Mat img2_no_nan = img2;
        cv::patchNaNs(img1_no_nan);
        cv::patchNaNs(img2_no_nan);
        cv::absdiff(img1_no_nan, img2_no_nan, diff);
    } else {
        cv::absdiff(img1, img2, diff);
    }
    diff = diff.reshape(1);

    return cv::countNonZero(diff) == 0;
}

}  // namespace robot::common::video
