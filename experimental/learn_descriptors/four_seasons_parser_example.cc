#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
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
        std::cout << "\nParser has " << parser.num_images() << " images!" << std::endl;
    }

    std::cout << "T_S_AS: \n"
              << parser.get_T_S_AS().matrix() << "\n\nT_cam_imu : \n\n"
              << parser.get_T_cam_imu().matrix() << "\n\nT_gps_imu: \n\n"
              << parser.get_T_gps_imu().matrix() << "\n\nT_e_gpsw: \n\n"
              << parser.get_T_e_gpsw().matrix() << "\n\ngnss scale: \n\n"
              << parser.get_gnss_scale() << std::endl;

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);
    cv::imshow("First + Last Images", img_first_and_last);

    for (size_t i = 0; i < parser.num_images(); i++) {
        cv::Mat img = parser.load_image(i);
        const std::string img_str = "img " + std::to_string(i);
        cv::putText(img, img_str, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        std::stringstream ss_gps;
        ss_gps << "GPS: ";
        if (img_pt.gps) {
            const Eigen::Vector3d& t = img_pt.gps->translation();
            ss_gps << t.x() << ", " << t.y() << ", " << t.z();
        } else {
            ss_gps << "N/A";
        }
        std::stringstream ss_ground_truth;
        ss_ground_truth << "Ground Truth: ";
        if (img_pt.ground_truth) {
            const Eigen::Vector3d& t = img_pt.ground_truth->translation();
            ss_ground_truth << t.x() << ", " << t.y() << ", " << t.z();
        } else {
            ss_ground_truth << "N/A";
        }
        cv::putText(img, ss_gps.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(img, ss_ground_truth.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        cv::imshow("FourSeasonsParserExample", img);
        int key = cv::waitKey(10);

        if (key == 'q' || key == 'Q') {
            break;
        }
    }
}