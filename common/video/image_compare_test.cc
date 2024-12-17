#include "common/video/image_compare.hh"

#include <gtest/gtest.h>

#include "opencv2/opencv.hpp"

namespace robot::common::video {

TEST(ImageCompareTest, ImagesEqual_SameImages) {
    cv::Mat img1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat img2 = img1.clone();
    EXPECT_TRUE(images_equal(img1, img2));
}

TEST(ImageCompareTest, ImagesEqual_DifferentSizes) {
    cv::Mat img1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat img2 = cv::Mat::zeros(200, 200, CV_8UC3);
    EXPECT_FALSE(images_equal(img1, img2));
}

TEST(ImageCompareTest, ImagesEqual_DifferentTypes) {
    cv::Mat img1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat img2 = cv::Mat::zeros(100, 100, CV_8UC1);
    EXPECT_FALSE(images_equal(img1, img2));
}

TEST(ImageCompareTest, ImagesEqual_DifferentContent) {
    cv::Mat img1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat img2 = cv::Mat::ones(100, 100, CV_8UC3);
    EXPECT_FALSE(images_equal(img1, img2));
}

}  // namespace robot::common::video