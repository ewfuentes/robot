#include "experimental/learn_descriptors/vo.hh"

#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors::vo {
TEST(VIO_TEST, frontend_pipeline_sweep) {
    const size_t width = 640;
    const size_t height = 480;

    const size_t pixel_shift_x = 20;
    const size_t PIXEL_COMP_TOL = 5;
    cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, pixel_shift_x, 0, 1, 0);
    cv::Mat image_1 = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat image_2;
    std::vector<cv::Point> points = {{50, 50},   {100, 100}, {150, 150}, {200, 200},
                                     {250, 250}, {300, 300}, {350, 350}, {50, 350}};
    for (const auto& point : points) {
        cv::circle(image_1, point, 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::warpAffine(image_1, image_2, translation_mat, image_1.size());

    cv::Mat img_test_disp;
    cv::hconcat(image_1, image_2, img_test_disp);
    cv::imshow("Test", img_test_disp);
    cv::waitKey(100);

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
    cv::Mat img_display_test;
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
            keypoints_descriptors_pair_1 = frontend.getKeypointsAndDescriptors(image_1);
            keypoints_descriptors_pair_2 = frontend.getKeypointsAndDescriptors(image_2);
            matches = frontend.getMatches(keypoints_descriptors_pair_1.second,
                                          keypoints_descriptors_pair_2.second);
            frontend.drawKeypoints(image_1, keypoints_descriptors_pair_1.first,
                                   img_keypoints_out_1);
            frontend.drawKeypoints(image_2, keypoints_descriptors_pair_2.first,
                                   img_keypoints_out_2);
            frontend.drawMatches(image_1, keypoints_descriptors_pair_1.first, image_2,
                                 keypoints_descriptors_pair_2.first, matches, img_matches_out);
            cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
            cv::vconcat(img_display_test, img_matches_out, img_display_test);
            cv::imshow("Keypoints and Matches Output", img_display_test);
            cv::waitKey(1000);
            printf("completed frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
            for (const cv::DMatch match : matches) {
                EXPECT_NEAR(keypoints_descriptors_pair_1.first[match.queryIdx].pt.x,
                            keypoints_descriptors_pair_2.first[match.trainIdx].pt.x,
                            PIXEL_COMP_TOL);
                EXPECT_NEAR(keypoints_descriptors_pair_1.first[match.queryIdx].pt.y,
                            keypoints_descriptors_pair_2.first[match.trainIdx].pt.y,
                            PIXEL_COMP_TOL);
            }
        }
    }
}
}  // namespace robot::experimental::learn_descriptors::vo
