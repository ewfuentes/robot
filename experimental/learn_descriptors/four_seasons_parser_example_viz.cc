#include <algorithm>
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

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    ROBOT_CHECK(parser.num_images() != 0);

    std::vector<Eigen::Isometry3d> w_from_gnss_cams;  // camera frames from visual world frame
    Eigen::Isometry3d scale_mat = Eigen::Isometry3d::Identity();
    std::cout << "gnss scale: " << parser.get_gnss_scale() << std::endl;
    scale_mat.linear() *= parser.get_gnss_scale();
    std::cout << "scale mat: " << scale_mat.matrix() << std::endl;
    for (size_t i = 100; i < std::min(static_cast<size_t>(800), parser.num_images()); i += 49) {
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        Eigen::Isometry3d AS_w_from_gnss_cam =
            scale_mat * Eigen::Isometry3d(img_pt.AS_w_from_gnss_cam->matrix());
        Eigen::Isometry3d w_from_gnss_cam =
            Eigen::Isometry3d(parser.get_S_from_AS().matrix()) * AS_w_from_gnss_cam;
        Eigen::Isometry3d w_from_vio_cam = Eigen::Isometry3d(parser.get_S_from_AS().matrix()) *
                                           Eigen::Isometry3d(img_pt.AS_w_from_vio_cam->matrix());
        w_from_gnss_cams.push_back(w_from_gnss_cam);
        std::cout << "\npose gnss metric " << i << w_from_gnss_cam.matrix() << std::endl;
        std::cout << "pose vio" << i << w_from_vio_cam.matrix() << std::endl;
    }
    std::cout << "got " << w_from_gnss_cams.size() << " poses" << std::endl;
    robot::geometry::viz_scene(w_from_gnss_cams, std::vector<Eigen::Vector3d>());
}