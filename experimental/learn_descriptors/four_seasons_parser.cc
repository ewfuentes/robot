#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/camera_calibration.hh"
#include "experimental/learn_descriptors/four_seasons_parser_detail.hh"

namespace robot::experimental::learn_descriptors {
using namespace detail::four_seasons_parser;
FourSeasonsParser::FourSeasonsParser(const std::filesystem::path& root_dir,
                                     const std::filesystem::path& calibration_dir)
    : root_dir_(root_dir),
      img_dir_(root_dir / "distorted_images" / "cam0"),
      cal_(txt_parser_help::load_camera_calibration(calibration_dir)),
      shared_transforms_(std::make_shared<FourSeasonsTransforms::StaticTransforms>(
          root_dir / "Transformations.txt")) {
    const std::filesystem::path path_img = root_dir_ / "times.txt";
    const std::filesystem::path path_gnss = root_dir_ / "GNSSPoses.txt";
    const std::filesystem::path path_vio = root_dir_ / "result.txt";
    const std::filesystem::path path_gps = root_dir_ / "septentrio.nmea";
    const size_t min_time_sig_figs = txt_parser_help::min_sig_figs_result_time(path_vio);
    std::cout << "heartbeat 0" << std::endl;
    txt_parser_help::TimeDataList img_time_list =
        txt_parser_help::create_img_time_data_list(path_img, min_time_sig_figs);
    std::cout << "heartbeat 1" << std::endl;
    txt_parser_help::TimeDataMap gnss_poses_time_map =
        txt_parser_help::create_gnss_poses_time_data_map(path_gnss, min_time_sig_figs);
    std::cout << "heartbeat 2" << std::endl;
    txt_parser_help::TimeDataMap vio_poses_time_map =
        txt_parser_help::create_vio_time_data_map(path_vio, min_time_sig_figs);
    std::cout << "heartbeat 3" << std::endl;
    gps_parser_help::TimeGPSList gps_time_list =
        gps_parser_help::create_gps_time_data_list(path_gps);
    std::cout << "heartbeat 4" << std::endl;

    size_t id = 0;
    for (const std::pair<size_t, std::vector<std::string>>& pair_time_data : img_time_list) {
        const size_t time_key = pair_time_data.first;
        ImagePointFourSeasons img_pt;
        img_pt.id = id;
        img_pt.K = cal_;
        std::cout << "heartbeat 5" << std::endl;
        img_pt.shared_static_transforms = shared_transforms_;
        img_pt.seq = std::stoull(
            pair_time_data.second[static_cast<size_t>(txt_parser_help::ImgIdx::TIME_NS)]);
        if (gnss_poses_time_map.find(time_key) != gnss_poses_time_map.end()) {
            const std::vector<std::string>& parsed_line_gnss_poses =
                gnss_poses_time_map.at(time_key);

            Eigen::Vector3d t_gps_cam_from_AS_w(
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_X)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_Y)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::TRAN_Z)]));
            Eigen::Quaterniond R_gps_cam_from_AS_w(
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_W)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_X)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_Y)]),
                std::stod(
                    parsed_line_gnss_poses[static_cast<size_t>(txt_parser_help::GPSIdx::QUAT_Z)]));
            img_pt.AS_w_from_gnss_cam = liegroups::SE3(R_gps_cam_from_AS_w, t_gps_cam_from_AS_w);
        } else {
            std::clog << "There is no AS_w_from_gnss_cam data at img_pt with id: " << id
                      << std::endl;
        }
        if (vio_poses_time_map.find(time_key) != vio_poses_time_map.end()) {
            const std::vector<std::string>& parsed_line_vio = vio_poses_time_map.at(time_key);
            Eigen::Vector3d t_AS_w_from_vio_cam(
                std::stod(parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_X)]),
                std::stod(parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_Y)]),
                std::stod(
                    parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::TRAN_Z)]));
            Eigen::Quaterniond R_AS_w_from_vio_cam(
                std::stod(parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_W)]),
                std::stod(parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_X)]),
                std::stod(parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_Y)]),
                std::stod(
                    parsed_line_vio[static_cast<size_t>(txt_parser_help::ResultIdx::QUAT_Z)]));
            img_pt.AS_w_from_vio_cam = liegroups::SE3(R_AS_w_from_vio_cam, t_AS_w_from_vio_cam);
        } else {
            std::clog << "There is no AS_w_from_vio_cam data at img_pt with id: " << id
                      << std::endl;
        }
        img_pt_vector_.push_back(img_pt);
        id++;
    }
    // popoulate gps to nearest img time
    // TODO: could be linear time... but good enough
    for (const auto& [time_unix_ns, gps_data] : gps_time_list) {
        auto it =
            std::lower_bound(img_pt_vector_.begin(), img_pt_vector_.end(), time_unix_ns,
                             [](const ImagePointFourSeasons& img_pt, const size_t query_unix_time) {
                                 return img_pt.seq < query_unix_time;
                             });
        size_t insert_idx = std::distance(img_pt_vector_.begin(), it);
        if (it != img_pt_vector_.begin() &&
            abs_diff(it->seq, time_unix_ns) > abs_diff(std::prev(it)->seq, time_unix_ns)) {
            insert_idx--;
        }
        // NOTE: in future, could perhaps use gps data that isn't associated with an img_pt in some
        // way. maybe to help with interpolation, estimate velocity
        if (abs_diff(img_pt_vector_[insert_idx].seq, time_unix_ns) <
            FourSeasonsParser::CAM_CAP_DELTA_NS) {
            img_pt_vector_[insert_idx].gps_gcs = gps_data;
        }
    }
}

cv::Mat FourSeasonsParser::load_image(const size_t m) const {
    return get_image_point(m).load_image(img_dir_);
}
}  // namespace robot::experimental::learn_descriptors