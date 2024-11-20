#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
TEST(SymphonyLakeParserTest, snippet_140106) {
    const std::filesystem::path image_root_dir = "external/symphony_lake_snippet/symphony_lake";
    const std::vector<std::string> survey_list{"140106_snippet"};

    DataParser data_parser = DataParser(image_root_dir, survey_list);

    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.get_surveys();

    cv::Mat image;
    cv::namedWindow("Symphony Dataset Image", cv::WINDOW_AUTOSIZE);
    printf("Press 'q' in graphic window to quit\n");
    for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
        const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
        for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
            image = survey.loadImageByImageIndex(j);
            cv::imshow("Symphony Dataset Image", image);
            cv::waitKey(2);
        }
    }
}
}  // namespace robot::experimental::learn_descriptors