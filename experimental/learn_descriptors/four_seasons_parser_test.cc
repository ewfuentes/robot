#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "common/check.hh"
#include "common/gps/frame_translation.hh"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/four_seasons_parser_detail.hh"
#include "gtest/gtest.h"
#include "nmea/message/gga.hpp"
#include "nmea/message/rmc.hpp"
#include "nmea/object/date.hpp"
#include "nmea/sentence.hpp"

static bool images_equal(cv::Mat img1, cv::Mat img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.reshape(1);
    return cv::countNonZero(diff) == 0;
}

namespace robot::experimental::learn_descriptors {
using namespace detail::four_seasons_parser;
TEST(FourSeasonsParserTest, parser_test) {
    const std::filesystem::path dir_snippet =
        "external/four_seasons_snippet/recording_2020-04-07_11-33-45";
    const std::filesystem::path dir_calibration = "external/four_seasons_snippet/calibration";

    EXPECT_TRUE(std::filesystem::exists(dir_snippet));
    EXPECT_TRUE(std::filesystem::exists(dir_calibration));

    std::cout << "heartbeat -1" << std::endl;
    FourSeasonsParser parser(dir_snippet, dir_calibration);

    // transformations test
    const liegroups::SE3 S_from_AS(Eigen::Quaterniond(0.999992, 0.000869, 0.003288, -0.002016),
                                   Eigen::Vector3d::Zero());
    const liegroups::SE3 cam_from_imu(Eigen::Quaterniond(-0.002350, -0.007202, 0.708623, -0.705546),
                                      Eigen::Vector3d(0.175412, 0.003689, -0.058106));
    const liegroups::SE3 w_from_gpsw(Eigen::Quaterniond(0.802501, 0.000344, -0.000541, 0.596650),
                                     Eigen::Vector3d(0.256671, 0.021142, 0.073751));
    const liegroups::SE3 gps_from_imu(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    const liegroups::SE3 e_from_gpsw(
        Eigen::Quaterniond(0.590438, 0.224931, 0.275937, 0.724326),
        Eigen::Vector3d(4164702.580389, 857109.771387, 4738828.771006));
    const double gnss_scale = 0.911501;
    EXPECT_TRUE(S_from_AS.matrix().isApprox(parser.S_from_AS().matrix()));
    EXPECT_TRUE(cam_from_imu.matrix().isApprox(parser.cam_from_imu().matrix()));
    EXPECT_TRUE(w_from_gpsw.matrix().isApprox(parser.w_from_gpsw().matrix()));
    EXPECT_TRUE(gps_from_imu.matrix().isApprox(parser.gps_from_imu().matrix()));
    EXPECT_TRUE(e_from_gpsw.matrix().isApprox(parser.e_from_gpsw().matrix()));
    EXPECT_DOUBLE_EQ(gnss_scale, parser.gnss_scale());

    // calibration test
    const std::shared_ptr<CameraCalibrationFisheye> calibration_target =
        txt_parser_help::load_camera_calibration(dir_calibration);
    const std::shared_ptr<CameraCalibrationFisheye> calibration = parser.camera_calibration();
    EXPECT_DOUBLE_EQ(calibration_target->cx, calibration->cx);
    EXPECT_DOUBLE_EQ(calibration_target->cy, calibration->cy);
    EXPECT_DOUBLE_EQ(calibration_target->fx, calibration->fx);
    EXPECT_DOUBLE_EQ(calibration_target->fy, calibration->fy);
    EXPECT_DOUBLE_EQ(calibration_target->k1, calibration->k1);
    EXPECT_DOUBLE_EQ(calibration_target->k2, calibration->k2);
    EXPECT_DOUBLE_EQ(calibration_target->k3, calibration->k3);
    EXPECT_DOUBLE_EQ(calibration_target->k4, calibration->k4);

    EXPECT_NE(parser.num_images(), 0);

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);

    const std::filesystem::path dir_img = dir_snippet / "distorted_images/cam0";
    const std::filesystem::path path_img_time = dir_snippet / "times.txt";
    const std::filesystem::path path_gnss_poses = dir_snippet / "GNSSPoses.txt";
    const std::filesystem::path path_vio = dir_snippet / "result.txt";
    const size_t min_sig_figs_time = txt_parser_help::min_sig_figs_result_time(path_vio);

    // fetch data for all the fields (AS_w_from_gnss_cam, img times, result)
    const txt_parser_help::TimeDataList img_time_list =
        txt_parser_help::create_img_time_data_list(path_img_time, min_sig_figs_time);
    EXPECT_EQ(img_time_list.size(), parser.num_images());
    const txt_parser_help::TimeDataMap gnss_poses_time_map =
        txt_parser_help::create_gnss_poses_time_data_map(path_gnss_poses, min_sig_figs_time);
    const txt_parser_help::TimeDataMap vio_time_map =
        txt_parser_help::create_vio_time_data_map(path_vio, min_sig_figs_time);

    Eigen::Matrix4d scale_mat = Eigen::Matrix4d::Identity();
    scale_mat(0, 0) = scale_mat(1, 1) = scale_mat(2, 2) = parser.gnss_scale();
    const Eigen::Isometry3d ECEF_from_AS_w = Eigen::Isometry3d(
        (parser.e_from_gpsw() * parser.w_from_gpsw().inverse() * parser.S_from_AS()).matrix() *
        scale_mat);

    std::vector<double> gps_ns_delta_from_shutter;

    for (size_t i = 0; i < parser.num_images(); i++) {
        const ImagePointFourSeasons& img_pt = parser.get_image_point(i);
        const cv::Mat img = parser.load_image(i);

        // seq (time in nanoseconds) test
        EXPECT_EQ(std::stod(img_time_list[i].second[0]), img_pt.seq);

        // image testimg_time_list[i].first
        const std::filesystem::path path_img = dir_img / (std::to_string(img_pt.seq) + ".png");
        const cv::Mat img_target = cv::imread(path_img);
        EXPECT_TRUE(images_equal(img_target, img));

        // check AS_w_from_gnss_cam and result entries
        if (gnss_poses_time_map.find(img_time_list[i].first) != gnss_poses_time_map.end()) {
            const std::vector<std::string>& parsed_line_gnss_poses =
                gnss_poses_time_map.at(img_time_list[i].first);

            Eigen::Vector3d gps_translation(
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_X)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_Y)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_Z)]));
            Eigen::Quaterniond gps_quat(
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_W)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_X)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_Y)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_Z)]));
            EXPECT_TRUE(liegroups::SE3(gps_quat, gps_translation)
                            .matrix()
                            .isApprox(img_pt.AS_w_from_gnss_cam->matrix()));
        } else if (img_pt.AS_w_from_gnss_cam) {
            throw std::runtime_error("img_pt has AS_w_from_gnss_cam point but files do not!");
        }
        if (vio_time_map.find(img_time_list[i].first) != vio_time_map.end()) {
            const std::vector<std::string>& parsed_line_gps =
                vio_time_map.at(img_time_list[i].first);
            Eigen::Vector3d ground_truth_translation(
                std::stod(parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_X)]),
                std::stod(parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_Y)]),
                std::stod(
                    parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_Z)]));
            Eigen::Quaterniond ground_truth_quat(
                std::stod(parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_W)]),
                std::stod(parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_X)]),
                std::stod(parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_Y)]),
                std::stod(
                    parsed_line_gps[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_Z)]));
            EXPECT_TRUE(liegroups::SE3(ground_truth_quat, ground_truth_translation)
                            .matrix()
                            .isApprox(img_pt.AS_w_from_vio_cam->matrix()));
        } else if (img_pt.AS_w_from_vio_cam) {
            throw std::runtime_error("img_pt has AS_w_from_vio_cam point but files do not!");
        }
        if (img_pt.AS_w_from_gnss_cam && img_pt.gps_gcs) {
            const Eigen::Isometry3d ECEF_from_gnss_cam(ECEF_from_AS_w.matrix() *
                                                       img_pt.AS_w_from_gnss_cam->matrix());
            const Eigen::Isometry3d gps_from_cam(
                (parser.gps_from_imu() * parser.cam_from_imu().inverse()).matrix());
            const Eigen::Isometry3d ECEF_from_gnss_gps =
                ECEF_from_gnss_cam * gps_from_cam.inverse();

            const Eigen::Vector3d& t_ECEF_from_gnss_gps = ECEF_from_gnss_gps.translation();
            const Eigen::Vector3d gnss_gps_gcs = gps::lla_from_ecef(t_ECEF_from_gnss_gps);

            const Eigen::Vector3d raw_gps_gcs(
                img_pt.gps_gcs->latitude, img_pt.gps_gcs->longitude,
                img_pt.gps_gcs->altitude ? *(img_pt.gps_gcs->altitude) : 0);

            EXPECT_NEAR(raw_gps_gcs.x(), gnss_gps_gcs.x(), 1e-4);  // latitude
            EXPECT_NEAR(raw_gps_gcs.y(), gnss_gps_gcs.y(), 1e-4);  // longitude
            EXPECT_NEAR(raw_gps_gcs.z(), gnss_gps_gcs.z(), 60);    // height above sea level
        }
        if (img_pt.gps_gcs) {
            if (img_pt.gps_gcs->seq > img_pt.seq) {
                gps_ns_delta_from_shutter.push_back(
                    static_cast<double>(img_pt.gps_gcs->seq - img_pt.seq));
            } else {
                gps_ns_delta_from_shutter.push_back(
                    -static_cast<double>(img_pt.seq - img_pt.gps_gcs->seq));
            }
        }
    }

    double max_delta =
        *std::max_element(gps_ns_delta_from_shutter.begin(), gps_ns_delta_from_shutter.end());
    ROBOT_CHECK(max_delta < FourSeasonsParser::CAM_CAP_DELTA_NS);
}
}  // namespace robot::experimental::learn_descriptors