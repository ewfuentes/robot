#include "experimental/learn_descriptors/visual_odometry.hh"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors {
TEST(VIO_TEST, frontend_pipeline_sweep) {
    const size_t width = 640;
    const size_t height = 480;

    const size_t pixel_shift_x = 20;
    const size_t PIXEL_COMP_TOL = 20;

    cv::Mat image_1 = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat image_2;

    cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, pixel_shift_x, 0, 1, 0);
    cv::Point rect_points[4] = {{150, 200}, {350, 200}, {350, 300}, {150, 300}};
    float com_x = 0.0f, com_y = 0.0f;
    for (const cv::Point& rect_point : rect_points) {
        com_x += rect_point.x;
        com_y += rect_point.y;
    }
    com_x *= 0.25;
    com_y *= 0.25;
    cv::Point rotation_center(com_x, com_y);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(rotation_center, 45, 1.0);

    const size_t line_spacing = 100;
    for (size_t i = 0; i <= width / line_spacing; i++) {
        size_t x = i * line_spacing + (width % line_spacing) / 2;
        size_t b = i * 255.0 / (width / line_spacing);
        size_t g = 255.0 - i * 255.0 / (width / line_spacing);
        cv::line(image_1, cv::Point(x, 0), cv::Point(x, height - 1), cv::Scalar(b, g, 0), 2);
    }
    for (size_t i = 0; i <= height / line_spacing; i++) {
        size_t y = i * line_spacing + (height % line_spacing) / 2;
        size_t b = i * 255.0 / (width / line_spacing);
        size_t g = 255.0 - i * 255.0 / (width / line_spacing);
        cv::line(image_1, cv::Point(0, y), cv::Point(width - 1, y), cv::Scalar(b, g, 0), 2);
    }

    cv::warpAffine(image_1, image_1, rotation_matrix, image_1.size());
    cv::warpAffine(image_1, image_2, translation_mat, image_1.size());

    // cv::Mat img_test_disp;
    // cv::hconcat(image_1, image_2, img_test_disp);
    // cv::imshow("Test", img_test_disp);
    // cv::waitKey(1000);

    Frontend::ExtractorType extractor_types[2] = {Frontend::ExtractorType::SIFT,
                                                  Frontend::ExtractorType::ORB};
    Frontend::MatcherType matcher_types[3] = {Frontend::MatcherType::BRUTE_FORCE,
                                              Frontend::MatcherType::FLANN,
                                              Frontend::MatcherType::KNN};

    Frontend frontend;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_1;
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> keypoints_descriptors_pair_2;
    std::vector<cv::DMatch> matches;
    cv::Mat img_keypoints_out_1(height, width, CV_8UC3),
        img_keypoints_out_2(height, width, CV_8UC3), img_matches_out(height, 2 * width, CV_8UC3);
    // cv::Mat img_display_test;
    for (Frontend::ExtractorType extractor_type : extractor_types) {
        for (Frontend::MatcherType matcher_type : matcher_types) {
            printf("started frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
            try {
                frontend = Frontend(extractor_type, matcher_type);
            } catch (const std::invalid_argument& e) {
                assert(std::string(e.what()) == "FLANN can not be used with ORB.");  // very jank...
                continue;
            }
            keypoints_descriptors_pair_1 = frontend.get_keypoints_and_descriptors(image_1);
            keypoints_descriptors_pair_2 = frontend.get_keypoints_and_descriptors(image_2);
            matches = frontend.get_matches(keypoints_descriptors_pair_1.second,
                                           keypoints_descriptors_pair_2.second);
            frontend.draw_keypoints(image_1, keypoints_descriptors_pair_1.first,
                                    img_keypoints_out_1);
            frontend.draw_keypoints(image_2, keypoints_descriptors_pair_2.first,
                                    img_keypoints_out_2);
            frontend.draw_matches(image_1, keypoints_descriptors_pair_1.first, image_2,
                                  keypoints_descriptors_pair_2.first, matches, img_matches_out);
            // cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
            // cv::vconcat(img_display_test, img_matches_out, img_display_test);
            // std::stringstream text;
            // text << "Extractor " << static_cast<int>(extractor_type) << ", matcher "
            //      << static_cast<int>(matcher_type);
            // cv::putText(img_display_test, text.str(), cv::Point(20, height - 50),
            //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            // cv::imshow("Keypoints and Matches Output.", img_display_test);
            // std::cout << "Press spacebar to pause." << std::endl;
            // while (cv::waitKey(1000) == 32) {
            // }
            printf("completed frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
            if (extractor_type != Frontend::ExtractorType::ORB) {  // don't check ORB for now
                for (const cv::DMatch match : matches) {
                    EXPECT_NEAR(keypoints_descriptors_pair_1.first[match.queryIdx].pt.x -
                                    keypoints_descriptors_pair_2.first[match.trainIdx].pt.x,
                                pixel_shift_x, pixel_shift_x + PIXEL_COMP_TOL);
                    EXPECT_NEAR(keypoints_descriptors_pair_2.first[match.trainIdx].pt.y -
                                    keypoints_descriptors_pair_1.first[match.queryIdx].pt.y,
                                0, PIXEL_COMP_TOL);
                }
            }
        }
    }
}
}  // namespace robot::experimental::learn_descriptors
