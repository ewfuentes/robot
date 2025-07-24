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

    const std::filesystem::path path_data = args["data_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    ROBOT_CHECK(parser.num_images() != 0);

    std::cout << "S_from_AS: \n"
              << parser.S_from_AS().matrix() << "\n\ncam_from_imu : \n\n"
              << parser.cam_from_imu().matrix() << "\n\ngps_from_imu: \n\n"
              << parser.w_from_gpsw().matrix() << "\n\nw_from_gpsw : \n\n"
              << parser.gps_from_imu().matrix() << "\n\ne_from_gpsw: \n\n"
              << parser.e_from_gpsw().matrix() << "\n\ngnss scale: \n\n"
              << parser.gnss_scale() << std::endl;

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);
    cv::imshow("First + Last Images", img_first_and_last);

    for (size_t i = 0; i < parser.num_images(); i += 1) {
        cv::Mat img = parser.load_image(i);
        const std::string img_str = "img " + std::to_string(i);
        cv::putText(img, img_str, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        const lrn_desc::ImagePointFourSeasons img_pt = parser.get_image_point(i);
        std::stringstream ss_AS_w_from_gnss_cam;
        ss_AS_w_from_gnss_cam << "AS_w_from_gnss_cam: ";
        if (img_pt.AS_w_from_gnss_cam) {
            const Eigen::Vector3d& t = img_pt.AS_w_from_gnss_cam->translation();
            ss_AS_w_from_gnss_cam << t.x() << ", " << t.y() << ", " << t.z();
        } else {
            ss_AS_w_from_gnss_cam << "N/A";
        }
        std::stringstream ss_AS_w_from_vio_cam;
        ss_AS_w_from_vio_cam << "AS_w_from_vio_cam: ";
        if (img_pt.AS_w_from_vio_cam) {
            const Eigen::Vector3d& t = img_pt.AS_w_from_vio_cam->translation();
            ss_AS_w_from_vio_cam << t.x() << ", " << t.y() << ", " << t.z();
        } else {
            ss_AS_w_from_vio_cam << "N/A";
        }
        std::stringstream ss_gps;
        ss_gps << "gps_gcs: ";
        if (img_pt.gps_gcs) {
            const lrn_desc::GPSData& gps_data = *img_pt.gps_gcs;
            ss_gps << "long: " << gps_data.longitude << ", lat: " << gps_data.latitude;
            if (gps_data.altitude) {
                ss_gps << ", alt: " << *gps_data.altitude;
            }
        } else {
            ss_gps << "N/A";
        }
        cv::putText(img, ss_AS_w_from_gnss_cam.str(), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, ss_AS_w_from_vio_cam.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, ss_gps.str(), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
        std::cout << img_pt.to_string() << std::endl;
        cv::imshow("FourSeasonsParserExample", img);
        int key = cv::waitKey(0);

        if (key == 'q' || key == 'Q') {
            break;
        }
    }
}