#include "experimental/learn_descriptors/symphony_lake_parser.hh"
#include "common/check.hh"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class SymphonyLakeDatasetTestHelper {
    public:
        static bool images_equal(cv::Mat img1, cv::Mat img2) {
            if (img1.size() != img2.size() || img1.type() != img2.type()) {
                return false;
            }
            cv::Mat diff;
            cv::absdiff(img1, img2, diff);
            diff = diff.reshape(1);
            return cv::countNonZero(diff) == 0;
        }
};
TEST(SymphonyLakeParserTest, snippet_140106) {
    const std::filesystem::path image_root_dir = "external/symphony_lake_snippet/symphony_lake";
    const std::vector<std::string> survey_list{"140106_snippet"};

    DataParser data_parser = DataParser(image_root_dir, survey_list);

    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();

    cv::Mat image;
    cv::Mat target_img;
    cv::namedWindow("Symphony Dataset Image", cv::WINDOW_AUTOSIZE);
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
            std::filesystem::path target_img_dir = image_root_dir / survey_list[i] / "0027" / target_img_name_str;
            target_img = cv::imread(target_img_dir.string());

            EXPECT_TRUE(SymphonyLakeDatasetTestHelper::images_equal(image, target_img));
            cv::imshow("Symphony Dataset Image", image);
            cv::waitKey(2);
        }
    }
}
}  // namespace robot::experimental::learn_descriptors