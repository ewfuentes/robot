#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors::symphony_lake_parser {
TEST(SymphonyLakeParserTest, hello_world) {
    hello_world("hello from symphony lake dataset. includes working well.");
}

TEST(SymphonyLakeParserTest, database_query) {
    std::string image_root_dir = "/home/pizzaroll04/Documents/datasets/symphony_lake_full";
    std::string surveys = "140106";
    std::vector<std::string> survey_list;
    std::cout << "\nEnter root image directory: ";
    // std::cin >> image_root_dir;
    std::cout << "\nEnter surveys separated by spaces: ";
    // std::getline(std::cin, surveys);

    std::string token;
    std::stringstream ss(surveys);
    while (std::getline(ss, token, ' ')) {
        survey_list.push_back(token);
    }

    DataParser data_parser = DataParser(image_root_dir, survey_list);

    const symphony_lake_dataset::SurveyVector &survey_vector = data_parser.getSurveys();

    cv::Mat image;
    cv::namedWindow("Symphony Dataset Image", cv::WINDOW_AUTOSIZE);
    printf("Press 'q' in graphic window to quit\n");
    for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
        const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
        for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
            int key = cv::waitKey(1) & 0xFF;
            switch (key) {
                case 'q':
                case 'Q':
                case 0x27:
                    return;
                    break;
                default:
                    break;
            }
            image = survey.loadImageByImageIndex(j);
            cv::imshow("Symphony Dataset Image", image);
        }
    }
}
}  // namespace robot::experimental::learn_descriptors::symphony_lake_parser