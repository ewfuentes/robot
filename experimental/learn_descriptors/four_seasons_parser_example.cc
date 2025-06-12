#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "experimental/learn_descriptors/four_seasons_parser.hh"

namespace lrn_desc = robot::experimental::learn_descriptors;

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    std::filesystem::path path_data, path_calibration;

    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        std::cout << arg << std::endl;
        if (arg == "--data_dir") {
            std::cout << "DATA DIR PROVIDED!" << std::endl;
            path_data = std::filesystem::path(std::string(argv[++i]));
        } else if (arg == "--calibration_dir") {
            std::cout << "CONFIG DIR PROVIDED!" << std::endl;
            path_calibration = std::filesystem::path(std::string(argv[++i]));
        }
    }

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    if (parser.num_images() == 0) {
        std::clog << "Warning: parser.num_images is 0! Terminating..." << std::endl;
    } else {
        std::cout << "Parser has " << parser.num_images() << " images!" << std::endl;
    }

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);
    cv::imshow("First + Last Images", img_first_and_last);

    for (size_t i = 0; i < parser.num_images(); i++) {
        cv::Mat img = parser.load_image(i);
        const std::string img_str = "img " + std::to_string(i);
        cv::putText(img, img_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 0), 2);
        cv::imshow("FourSeasonsParserExample", img);
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        std::cout << "\rGPS: " << img_pt.gps.translation()
                  << "\tGround Truth: " << img_pt.ground_truth.matrix() << std::flush;

        int key = cv::waitKey(10);

        if (key == 'q' || key == 'Q') {
            break;
        }
    }
}