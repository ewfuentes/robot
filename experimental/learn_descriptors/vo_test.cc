#include "experimental/learn_descriptors/vo.hh"

#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors::vo {
TEST(VIO_TEST, frontend_pipeline_sweep) {
    int width = 640;
    int height = 480;

    cv::Mat random_image_1(height, width, CV_8UC3), random_image_2(height, width, CV_8UC3);
    cv::randu(random_image_1, cv::Scalar::all(0), cv::Scalar::all(256));
    cv::randu(random_image_2, cv::Scalar::all(0), cv::Scalar::all(256));

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
            } catch (const std::invalid_argument &e) {
                assert(std::string(e.what()) == "FLANN can not be used with ORB.");  // very jank...
                continue;
            }
            keypoints_descriptors_pair_1 = frontend.getKeypointsAndDescriptors(random_image_1);
            keypoints_descriptors_pair_2 = frontend.getKeypointsAndDescriptors(random_image_2);
            matches = frontend.getMatches(keypoints_descriptors_pair_1.second,
                                          keypoints_descriptors_pair_2.second);
            frontend.drawKeypoints(random_image_1, keypoints_descriptors_pair_1.first,
                                   img_keypoints_out_1);
            frontend.drawKeypoints(random_image_2, keypoints_descriptors_pair_2.first,
                                   img_keypoints_out_2);
            frontend.drawMatches(random_image_1, keypoints_descriptors_pair_1.first, random_image_2,
                                 keypoints_descriptors_pair_2.first, matches, img_matches_out);
            cv::hconcat(img_keypoints_out_1, img_keypoints_out_2, img_display_test);
            cv::vconcat(img_display_test, img_matches_out, img_display_test);
            cv::imshow("Keypoints and Matches Output", img_display_test);
            cv::waitKey(100);
            printf("completed frontend combination: (%d, %d)\n", static_cast<int>(extractor_type),
                   static_cast<int>(matcher_type));
        }
    }
}
}  // namespace robot::experimental::learn_descriptors::vo
