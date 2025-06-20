#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/check.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "visualization/opencv/opencv_viz.hh"

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

    const std::filesystem::path path_data = args["data_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();

    std::cout << "Hmm" << std::endl;

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    ROBOT_CHECK(parser.num_images() != 0);

    std::cout << "S_from_AS: \n"
              << parser.get_S_from_AS().matrix() << "\n\ncam_from_imu : \n\n"
              << parser.get_cam_from_imu().matrix() << "\n\ngps_from_imu: \n\n"
              << parser.get_w_from_gpsw().matrix() << "\n\nw_from_gpsw : \n\n"
              << parser.get_gps_from_imu().matrix() << "\n\ne_from_gpsw: \n\n"
              << parser.get_e_from_gpsw().matrix() << "\n\ngnss scale: \n\n"
              << parser.get_gnss_scale() << std::endl;

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);
    cv::imshow("First + Last Images", img_first_and_last);

    std::vector<Eigen::Isometry3d> poses_gps_from_;
    for (size_t i = 0; i < parser.num_images(); i++) {
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
    }
}