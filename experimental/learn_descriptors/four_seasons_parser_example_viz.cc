#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <numeric>
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

    std::vector<robot::geometry::VizPose> viz_frames;   // camera frames from visual world frame
    std::vector<robot::geometry::VizPoint> viz_points;  // camera frames from visual world frame

    Eigen::Isometry3d scale_mat = Eigen::Isometry3d::Identity();
    std::cout << "gnss scale: " << parser.gnss_scale() << std::endl;
    scale_mat.linear() *= parser.gnss_scale();
    std::cout << "scale mat: " << scale_mat.matrix() << std::endl;
    std::optional<double> altitude_gps_from_gnss_cam;
    std::vector<double> gps_ns_delta_from_shutter;
    for (size_t i = 0; i < parser.num_images(); i += 1) {
        const lrn_desc::ImagePoint img_pt = parser.get_image_point(i);
        std::cout << img_pt.to_string() << std::endl;
        if (!img_pt.AS_w_from_gnss_cam) {
            continue;
        }
        Eigen::Isometry3d w_from_gnss_cam(Eigen::Isometry3d(parser.S_from_AS().matrix()) *
                                          scale_mat *
                                          Eigen::Isometry3d(img_pt.AS_w_from_gnss_cam->matrix()));

        viz_frames.emplace_back(w_from_gnss_cam, "x_ref_" + std::to_string(i));
        viz_frames.emplace_back(w_from_gnss_cam, "x_ref_" + std::to_string(i));
        if (img_pt.gps_gcs) {
            if (img_pt.gps_gcs->seq > img_pt.seq) {
                gps_ns_delta_from_shutter.push_back(
                    static_cast<double>(img_pt.gps_gcs->seq - img_pt.seq));
            } else {
                gps_ns_delta_from_shutter.push_back(
                    -static_cast<double>(img_pt.seq - img_pt.gps_gcs->seq));
            }
            Eigen::Vector3d gcs_coordinate(
                img_pt.gps_gcs->latitude, img_pt.gps_gcs->longitude,
                img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0);

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
            Eigen::Vector4d gps_in_w =
                Eigen::Matrix4d((parser.w_from_gpsw() * parser.e_from_gpsw().inverse()).matrix()) *
                ECEF_from_gps_hom;
            viz_points.emplace_back(gps_in_w.head<3>(), "x_gps_" + std::to_string(i));
            const Eigen::Vector3d gps_from_ref_in_world =
                gps_in_w.head<3>() - w_from_gnss_cam.translation();
            std::cout << "gps_from_ref_in_world: " << gps_from_ref_in_world << std::endl;
        }
    }
    const double sum =
        std::accumulate(gps_ns_delta_from_shutter.begin(), gps_ns_delta_from_shutter.end(), 0.0);
    double avg_ns_gps_delta = sum / gps_ns_delta_from_shutter.size();
    double var_ns_gps_delta;
    double var_sum = 0.0;
    size_t num_greater_cam_hz = 0;

    for (const double delta : gps_ns_delta_from_shutter) {
        var_sum += std::pow(delta - avg_ns_gps_delta, 2);
        if (delta > lrn_desc::FourSeasonsParser::CAM_CAP_DELTA_NS) {
            num_greater_cam_hz++;
        }
    }
    var_ns_gps_delta = var_sum / gps_ns_delta_from_shutter.size();
    double max_delta =
        *std::max_element(gps_ns_delta_from_shutter.begin(), gps_ns_delta_from_shutter.end());
    ROBOT_CHECK(max_delta < lrn_desc::FourSeasonsParser::CAM_CAP_DELTA_NS);
    std::cout << "\nGPS Analysis: " << std::endl;
    std::cout << "\tavg delta time ns img_pt_seq from gps_gcs_seq: " << avg_ns_gps_delta
              << std::endl;
    std::cout << "\tstd_var delta time ns img_pt_seq from gps_gcs_seq: "
              << std::pow(var_ns_gps_delta, 0.5) << std::endl;
    std::cout << "\tnum deltas greater than cam_hz: " << num_greater_cam_hz << std::endl;
    std::cout << "max delta: " << max_delta << std::endl;
    std::cout << "\ngot " << viz_frames.size() << " poses" << std::endl;
    std::cout << "got " << viz_points.size() << " points" << std::endl;
    robot::geometry::viz_scene(viz_frames, viz_points, cv::viz::Color::brown(), true, true,
                               "Viz Trajectory + GPS Points");
}
