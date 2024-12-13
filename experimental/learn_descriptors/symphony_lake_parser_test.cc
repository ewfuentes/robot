#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/check.hh"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
namespace {
bool is_test() { return std::getenv("BAZEL_TEST") != nullptr; }
}  // namespace
TEST(SymphonyLakeParserTest, snippet_140106) {
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();

    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();

    cv::Mat image;
    cv::Mat target_img;
    if (!is_test()) {
        cv::namedWindow("Symphony Dataset Image", cv::WINDOW_AUTOSIZE);
    }
    printf("Press 'q' in graphic window to quit\n");
    for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
        const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
        for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
            image = survey.loadImageByImageIndex(j);

            // get the target image
            std::stringstream target_img_name;
            target_img_name << "0000" << j << ".jpg";
            const size_t target_img_name_length = 8;
            std::string target_img_name_str = target_img_name.str();
            target_img_name_str.replace(0, target_img_name_str.size() - target_img_name_length, "");
            std::filesystem::path target_img_dir =
                SymphonyLakeDatasetTestHelper::get_test_iamge_root_dir() / SymphonyLakeDatasetTestHelper::get_test_survey_list()[i] / "0027" / target_img_name_str;
            target_img = cv::imread(target_img_dir.string());

            EXPECT_TRUE(SymphonyLakeDatasetTestHelper::images_equal(image, target_img));
            // if (!is_test()) {
            //     cv::imshow("Symphony Dataset Image", image);
            //     cv::waitKey(2);
            // }
            // cv::imshow("Symphony Dataset Image", image);
            // cv::waitKey(200);
        }
    }
}
TEST(SymphonyLakeParserTest, snippet_140106_generator_test) {
    DataParser data_parser = SymphonyLakeDatasetTestHelper::get_test_parser();
    DataParser::Generator<cv::Mat> generator = (SymphonyLakeDatasetTestHelper::get_test_parser()).create_img_generator();
    DataParser::Generator<cv::Mat>::iterator img_iter = generator.begin();
    while (img_iter != generator.end()) {
        cv::Mat img = *img_iter;
        // if (!is_test()) {
        //     cv::imshow("Symphony Dataset Image", img);
        //     cv::waitKey(200);
        // }
        // cv::imshow("Symphony Dataset Image", img);
        // cv::waitKey(200);
        ++img_iter;
    }
}
}  // namespace robot::experimental::learn_descriptors
