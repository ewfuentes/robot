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
    std::vector<robot::geometry::VizPose> viz_frames;   // camera frames from visual world frame
    std::vector<robot::geometry::VizPoint> viz_points;  // camera frames from visual world frame

    Eigen::Isometry3d scale_mat = Eigen::Isometry3d::Identity();
    std::cout << "gnss scale: " << parser.gnss_scale() << std::endl;
    scale_mat.linear() *= parser.gnss_scale();
    std::cout << "scale mat: " << scale_mat.matrix() << std::endl;
    constexpr size_t START = 100;
    const size_t END = std::min(static_cast<size_t>(800), parser.num_images());
    std::optional<double> altitude_gps_from_gnss_cam;
    for (size_t i = START; i < END; i += 49) {
        std::cout << "\nImage point " << i << std::endl;
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        std::cout << "Image point seq: " << img_pt.seq << std::endl;
        if (!img_pt.AS_w_from_gnss_cam) {
            continue;
        }
        Eigen::Isometry3d AS_w_from_gnss_cam =
            scale_mat * Eigen::Isometry3d(img_pt.AS_w_from_gnss_cam->matrix());
        // Eigen::Isometry3d(img_pt.AS_w_from_gnss_cam->matrix());
        std::cout << "img_pt.AS_w_from_gnss_cam->matrix()" << img_pt.AS_w_from_gnss_cam->matrix()
                  << std::endl;
        std::cout << "AS_w_from_gnss_cam: " << AS_w_from_gnss_cam.matrix() << std::endl;
        Eigen::Isometry3d w_from_gnss_cam =
            Eigen::Isometry3d(parser.S_from_AS().matrix()) * AS_w_from_gnss_cam;
        w_from_gnss_cam.translation().x() += 1.2;
        w_from_gnss_cam.translation().y() += 1.2;
        // Eigen::Isometry3d w_from_vio_cam = Eigen::Isometry3d(parser.S_from_AS().matrix()) *
        //                                    Eigen::Isometry3d(img_pt.AS_w_from_vio_cam->matrix());
        std::cout << "w_from_gnss_cam: " << w_from_gnss_cam.matrix() << std::endl;
        viz_frames.emplace_back(w_from_gnss_cam, "x_ref_" + std::to_string(i));
        viz_frames.emplace_back(w_from_gnss_cam, "x_ref_" + std::to_string(i));
        if (i == START) {
            break;
            viz_points.emplace_back(w_from_gnss_cam.translation(), "x_gps_" + std::to_string(i));
        } else if (img_pt.gps_gcs) {
            std::cout << "gps time: " << img_pt.gps_gcs->seq << std::endl;
            Eigen::Vector3d gcs_coordinate(
                img_pt.gps_gcs->latitude, img_pt.gps_gcs->longitude,
                img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0);

            std::cout << "gcs_coordinate altitude: "
                      << (img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0) << std::endl;

            // compare the reference in gcs to the gps in gcs
            Eigen::Isometry3d ECEF_from_gnss_cam =
                Eigen::Isometry3d(
                    (parser.e_from_gpsw() * parser.w_from_gpsw().inverse()).matrix()) *
                w_from_gnss_cam;
            const Eigen::Vector3d gnss_cam_in_gcs =
                parser.gcs_from_ECEF(ECEF_from_gnss_cam.translation());

            if (!altitude_gps_from_gnss_cam) {
                altitude_gps_from_gnss_cam = gnss_cam_in_gcs.z() - *(img_pt.gps_gcs->altitude);
            }
            gcs_coordinate.z() += *altitude_gps_from_gnss_cam;

            std::cout << std::setprecision(20) << "gnss_cam_in_gcs: " << gnss_cam_in_gcs
                      << "\ngps_gcs: " << gcs_coordinate
                      << "\ngps-gnss_cam: " << (gcs_coordinate - gnss_cam_in_gcs) << std::endl;

            const Eigen::Vector3d ECEF_from_gps = parser.ECEF_from_gcs(gcs_coordinate);
            const Eigen::Vector4d ECEF_from_gps_hom(ECEF_from_gps.x(), ECEF_from_gps.y(),
                                                    ECEF_from_gps.z(), 1);
            // Eigen::Isometry3d w_from_gcs_gps =
            Eigen::Vector4d gps_in_w =
                Eigen::Matrix4d((parser.w_from_gpsw() * parser.e_from_gpsw().inverse()).matrix()) *
                ECEF_from_gps_hom;
            viz_points.emplace_back(gps_in_w.head<3>(), "x_gps_" + std::to_string(i));
            const Eigen::Vector3d gps_from_ref_in_world =
                gps_in_w.head<3>() - w_from_gnss_cam.translation();
            std::cout << "gps_from_ref_in_world: " << gps_from_ref_in_world << std::endl;
        }
        // std::cout << "\npose gnss metric " << i << w_from_gnss_cam.matrix() << std::endl;
        // std::cout << "pose vio" << i << w_from_vio_cam.matrix() << std::endl;
    }
    std::cout << "\ngot " << viz_frames.size() << " poses" << std::endl;
    std::cout << "got " << viz_points.size() << " points" << std::endl;
    robot::geometry::viz_scene(viz_frames, viz_points, cv::viz::Color::brown());
}