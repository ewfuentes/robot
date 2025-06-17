#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "common/check.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/four_seasons_parser.hh"

namespace lrn_desc = robot::experimental::learn_descriptors;

int main(int argc, const char** argv) {
    // clang-format off
    cxxopts::Options options("four_seasons_parser_example", "Demonstrate usage of four_seasons_parser");
    options.add_options()
        ("data_dir", "Path to dataset root directory", cxxopts::value<std::string>())
        ("calibration_dir", "Path to dataset calibration directory", cxxopts::value<std::string>())
        ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    const auto check_required = [&](const std::string& opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    check_required("data_dir");
    check_required("calibration_dir");

    const std::filesystem::path path_data = args["config_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    ROBOT_CHECK(parser.num_images() != 0);

    std::cout << "AS_from_S: \n"
              << parser.get_AS_from_S().matrix() << "\n\nimu_from_cam : \n\n"
              << parser.get_imu_from_cam().matrix() << "\n\nimu_from_gps: \n\n"
              << parser.get_gpsw_from_w().matrix() << "\n\ngpsw_from_w : \n\n"
              << parser.get_imu_from_gps().matrix() << "\n\ngpsw_from_e: \n\n"
              << parser.get_gpsw_from_e().matrix() << "\n\ngnss scale: \n\n"
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