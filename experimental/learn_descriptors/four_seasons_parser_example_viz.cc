#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/check.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "opencv2/opencv.hpp"
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

    // std::vector<robot::geometry::VizPose>
    //     w_from_gnss_cams;                                   // camera frames from visual world
    //     frame
    // std::vector<robot::geometry::VizPose> w_from_gcs_cams;  // camera frames from visual world
    // frame
    std::vector<robot::geometry::VizPose> viz_frames;  // camera frames from visual world frame

    Eigen::Isometry3d scale_mat = Eigen::Isometry3d::Identity();
    std::cout << "gnss scale: " << parser.gnss_scale() << std::endl;
    scale_mat.linear() *= parser.gnss_scale();
    std::cout << "scale mat: " << scale_mat.matrix() << std::endl;
    constexpr size_t START = 100;
    const size_t END = std::min(static_cast<size_t>(800), parser.num_images());
    for (size_t i = START; i < END; i += 49) {
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        Eigen::Isometry3d AS_w_from_gnss_cam =
            scale_mat * Eigen::Isometry3d(img_pt.AS_w_from_gnss_cam->matrix());
        Eigen::Isometry3d w_from_gnss_cam =
            Eigen::Isometry3d(parser.S_from_AS().matrix()) * AS_w_from_gnss_cam;
        Eigen::Vector3d
            // Eigen::Isometry3d w_from_vio_cam = Eigen::Isometry3d(parser.S_from_AS().matrix()) *
            //                                    Eigen::Isometry3d(img_pt.AS_w_from_vio_cam->matrix());
            viz_frames.emplace_back(w_from_gnss_cam, "x_ref_" + std::to_string(i));
        if (i == START) {
            viz_frames.emplace_back(w_from_gnss_cam, "x_gps_" + std::to_string(i));
        } else if (img_pt.gps_gcs) {
            const Eigen::Vector3d gcs_coordinate(
                img_pt.gps_gcs->latitude, img_pt.gps_gcs->longitude,
                img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0);
            const Eigen::Vector3d ECEF_from_gps = parser.ECEF_from_gcs(gcs_coordinate);
            const Eigen::Vector4d ECEF_from_gps_hom(ECEF_from_gps.x(), ECEF_from_gps.y(),
                                                    ECEF_from_gps.z(), 1);
            // Eigen::Isometry3d w_from_gcs_gps =
            Eigen::Vector4d gps_in_w =
                Eigen::Matrix4d((parser.w_from_gpsw() * parser.e_from_gpsw().inverse()).matrix()) *
                ECEF_from_gps_hom;
            Eigen::Isometry3d w_from_gcs_gps;
            w_from_gcs_gps.translation() = gps_in_w.head<3>();
            viz_frames.emplace_back(w_from_gcs_gps, "x_gps_" + std::to_string(i));
            const Eigen::Vector3d ref_to_gps_in_world =
                w_from_gcs_gps.translation() - w_from_gnss_cam.translation();
            std::cout << "ref_" << i << "_to_gps_in_world: " << ref_to_gps_in_world << std::endl;
        }
        // std::cout << "\npose gnss metric " << i << w_from_gnss_cam.matrix() << std::endl;
        // std::cout << "pose vio" << i << w_from_vio_cam.matrix() << std::endl;
    }
    std::cout << "got " << viz_frames.size() << " poses" << std::endl;
    robot::geometry::viz_scene(viz_frames, std::vector<robot::geometry::VizPoint>(),
                               cv::viz::Color::brown());
}