#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "GeographicLib/Geocentric.hpp"
#include "GeographicLib/LocalCartesian.hpp"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "common/check.hh"
#include "common/liegroups/se3.hh"
#include "common/time/robot_time.hh"
#include "nmea/message/gga.hpp"
#include "nmea/message/rmc.hpp"
#include "nmea/sentence.hpp"

namespace robot::experimental::learn_descriptors {
using namespace detail;
FourSeasonsParser::FourSeasonsParser(const std::filesystem::path& root_dir,
                                     const std::filesystem::path& calibration_dir)
    : root_dir_(root_dir),
      img_dir_(root_dir / "distorted_images" / "cam0"),
      cal_(txt_parser_help::load_camera_calibration(calibration_dir)),
      transforms_(root_dir / "Transformations.txt") {
    const std::filesystem::path path_img = root_dir_ / "times.txt";
    const std::filesystem::path path_gnss = root_dir_ / "GNSSPoses.txt";
    const std::filesystem::path path_vio = root_dir_ / "result.txt";
    const std::filesystem::path path_gps = root_dir_ / "septentrio.nmea";
    const size_t min_time_sig_figs = txt_parser_help::min_sig_figs_result_time(path_vio);
    txt_parser_help::TimeDataList img_time_list =
        txt_parser_help::create_img_time_data_list(path_img, min_time_sig_figs);
    txt_parser_help::TimeDataMap gnss_poses_time_map =
        txt_parser_help::create_gnss_poses_time_data_map(path_gnss, min_time_sig_figs);
    txt_parser_help::TimeDataMap vio_poses_time_map =
        txt_parser_help::create_vio_time_data_map(path_vio, min_time_sig_figs);
    gps_parser_help::TimeGPSList gps_time_list =
        gps_parser_help::create_gps_time_data_list(path_gps);

    size_t id = 0;
    for (const std::pair<size_t, std::vector<std::string>>& pair_time_data : img_time_list) {
        const size_t time_key = pair_time_data.first;
        ImagePoint img_pt;
        img_pt.id = id;
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
        auto it = std::lower_bound(img_pt_vector_.begin(), img_pt_vector_.end(), time_unix_ns,
                                   [](const ImagePoint& img_pt, const size_t query_unix_time) {
                                       return img_pt.seq < query_unix_time;
                                   });
        size_t insert_idx = std::distance(img_pt_vector_.begin(), it);
        if (it != img_pt_vector_.begin() &&
            detail::abs_diff(it->seq, time_unix_ns) >
                detail::abs_diff(std::prev(it)->seq, time_unix_ns)) {
            insert_idx--;
        }
        // NOTE: in future, could perhaps use gps data that isn't associated with an img_pt in some
        // way. maybe to help with interpolation, estimate velocity
        if (detail::abs_diff(img_pt_vector_[insert_idx].seq, time_unix_ns) <
            FourSeasonsParser::CAM_CAP_DELTA_NS) {
            img_pt_vector_[insert_idx].gps_gcs = gps_data;
        }
    }
}

cv::Mat FourSeasonsParser::load_image(const size_t m) const {
    return get_image_point(m).load_image(img_dir_);
}

FourSeasonsParser::FourSeasonsTransforms::FourSeasonsTransforms(
    const std::filesystem::path& path_transforms) {
    std::ifstream file_transforms(path_transforms);
    std::string line;
    while (std::getline(file_transforms, line)) {
        if (line.find("transform_S_AS") != std::string::npos) {
            std::getline(file_transforms, line);
            S_from_AS = get_transform_from_line(line);
        } else if (line.find("TS_cam_imu") != std::string::npos) {
            std::getline(file_transforms, line);
            cam_from_imu = get_transform_from_line(line);
        } else if (line.find("transform_w_gpsw") != std::string::npos) {
            std::getline(file_transforms, line);
            w_from_gpsw = get_transform_from_line(line);
        } else if (line.find("transform_gps_imu") != std::string::npos) {
            std::getline(file_transforms, line);
            gps_from_imu = get_transform_from_line(line);
        } else if (line.find("transform_e_gpsw") != std::string::npos) {
            std::getline(file_transforms, line);
            e_from_gpsw = get_transform_from_line(line);
        } else if (line.find("GNSS scale") != std::string::npos) {
            std::getline(file_transforms, line);
            gnss_scale = std::stod(line);
        }
    }
}

liegroups::SE3 FourSeasonsParser::FourSeasonsTransforms::get_transform_from_line(
    const std::string& line) {
    enum TransformEntry { T_X, T_Y, T_Z, Q_X, Q_Y, Q_Z, Q_W };
    std::vector<std::string> parsed_transform_line = txt_parser_help::parse_line_adv(line, ",");
    if (parsed_transform_line.size() < 7) {
        std::stringstream error_stream;
        error_stream << "parsed_transform_line doesn't have sufficient entries for "
                        "transform! parsed_transform_line.size(): "
                     << parsed_transform_line.size() << std::endl;
        throw std::runtime_error(error_stream.str());
    }
    std::vector<double> transform_nums;
    for (const std::string& num : parsed_transform_line) {
        transform_nums.push_back(static_cast<double>(std::stod(num)));
    }
    Eigen::Vector3d translation(transform_nums[TransformEntry::T_X],
                                transform_nums[TransformEntry::T_Y],
                                transform_nums[TransformEntry::T_Z]);
    Eigen::Quaterniond rotation(
        transform_nums[TransformEntry::Q_W], transform_nums[TransformEntry::Q_X],
        transform_nums[TransformEntry::Q_Y], transform_nums[TransformEntry::Q_Z]);
    return liegroups::SE3(rotation, translation);
}

Eigen::Vector3d FourSeasonsParser::gcs_from_ECEF(const Eigen::Vector3d& t_place_from_ECEF) {
    static const GeographicLib::Geocentric earth = GeographicLib::Geocentric::WGS84();

    double x = t_place_from_ECEF.x(), y = t_place_from_ECEF.y(), z = t_place_from_ECEF.z();

    double lat_deg, lon_deg, alt_m;
    earth.Reverse(x, y, z, lat_deg, lon_deg, alt_m);

    return Eigen::Vector3d(lat_deg, lon_deg, alt_m);
}

Eigen::Vector3d FourSeasonsParser::ECEF_from_gcs(const Eigen::Vector3d& gcs_coordinate) {
    double lat_deg = gcs_coordinate.x();
    double lon_deg = gcs_coordinate.y();
    double alt_m = gcs_coordinate.z();

    double x, y, z;

    static const GeographicLib::Geocentric earth = GeographicLib::Geocentric::WGS84();
    earth.Forward(lat_deg, lon_deg, alt_m, x, y, z);

    return Eigen::Vector3d(x, y, z);
}

namespace detail {
namespace txt_parser_help {

std::vector<std::string> parse_line_adv(const std::string& line, const std::string& delim) {
    if (delim == " ") {
        return std::vector<std::string>(
            absl::StrSplit(line, absl::ByAnyChar(" \t\n\r"), absl::SkipWhitespace()));
    }
    return std::vector<std::string>(absl::StrSplit(line, delim, absl::SkipWhitespace()));
}

FourSeasonsParser::CameraCalibrationFisheye load_camera_calibration(
    const std::filesystem::path& calibration_dir) {
    std::ifstream file_calibration(calibration_dir / "calib_0.txt");
    std::string line_calibration;
    std::getline(file_calibration, line_calibration);
    std::vector<std::string> parsed_calib_line = parse_line_adv(line_calibration, " ");
    return FourSeasonsParser::CameraCalibrationFisheye{
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FX)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FY)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CX)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CY)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K1)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K2)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K3)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K4)])};
}

// minimum sig figs of time in seconds (of type double) for all entries of results.txt
size_t min_sig_figs_result_time(const std::filesystem::path& path_vio) {
    std::ifstream file_vio(path_vio);
    std::string line;
    int min_figs = INT_MAX;
    while (std::getline(file_vio, line)) {
        const std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        const std::string str_time_seconds = parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)];
        const int figs =
            str_time_seconds.size() +
            (str_time_seconds.find('.') != std::string::npos ? -1 : 0);  // '.' is not a fig
        min_figs = std::min(min_figs, figs);
    }
    return min_figs;
}

const TimeDataList create_img_time_data_list(const std::filesystem::path& path_img,
                                             const size_t time_sig_figs) {
    TimeDataList time_map_img;
    std::ifstream file_img(path_img);
    std::string line;
    while (std::getline(file_img, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        size_t time_ns = std::stoull(parsed_line[static_cast<size_t>(ImgIdx::TIME_NS)]);
        time_map_img.push_back(
            std::make_pair(round_to_sig_figs(time_ns, time_sig_figs), parsed_line));
    }
    return time_map_img;
}

const TimeDataMap create_gnss_poses_time_data_map(const std::filesystem::path& path_gnss,
                                                  const size_t time_sig_figs) {
    TimeDataMap time_map_gnss_poses;
    std::ifstream file_gnss_poses(path_gnss);
    std::string line;
    std::getline(file_gnss_poses,
                 line);  // advance a line to get past the top comment in GNSSPoses.txt
    while (std::getline(file_gnss_poses, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, ",");
        size_t time_ns = std::stoull(parsed_line[static_cast<size_t>(GPSIdx::TIME_NS)]);
        time_map_gnss_poses.insert({round_to_sig_figs(time_ns, time_sig_figs), parsed_line});
    }
    return time_map_gnss_poses;
}

const TimeDataMap create_vio_time_data_map(const std::filesystem::path& path_vio,
                                           const size_t time_sig_figs) {
    TimeDataMap time_map_vio;
    std::ifstream file_vio(path_vio);
    std::string line;
    while (std::getline(file_vio, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        double time_seconds = std::stod(parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)]);
        size_t time_ns = time_seconds * 1e9;
        time_map_vio.insert({round_to_sig_figs(time_ns, time_sig_figs), parsed_line});
    }
    return time_map_vio;
}
}  // namespace txt_parser_help

namespace gps_parser_help {
size_t gps_utc_to_unix_time(const nmea::date& utc_date, const double utc_time_day_seconds) {
    std::chrono::sys_days date = std::chrono::year_month_day{
        std::chrono::year{utc_date.year + 2000}, std::chrono::month{utc_date.month},
        std::chrono::day{utc_date.day}};
    std::chrono::nanoseconds utc_time_day_ns =
        std::chrono::nanoseconds{static_cast<size_t>(utc_time_day_seconds * 1e9)};
    std::chrono::sys_time<std::chrono::nanoseconds> timestamp = date + utc_time_day_ns;
    return timestamp.time_since_epoch().count();
}
TimeGPSList create_gps_time_data_list(const std::filesystem::path& path_gps) {
    TimeGPSList time_list_gps;
    std::ifstream file_gps(path_gps);
    std::string line;
    std::optional<nmea::sentence> nmea_sentence;
    nmea::date date_last;
    date_last.day = 255;
    date_last.month = 255;
    date_last.year = 255;
    double time_of_day_last = -1.0;
    while (std::getline(file_gps, line) && !line.empty()) {
        try {
            nmea_sentence = nmea::sentence(line);
        } catch (const std::exception& e) {
            std::cerr << "failed to parse line as nmea sentence: " << e.what()
                      << "\ncontinuing to next line\n";
            continue;
        }
        if (nmea_sentence->type() == "GGA") {
            nmea::gga gga(*nmea_sentence);
            if (gga.utc.exists() && gga.latitude.exists() && gga.longitude.exists()) {
                if (std::abs(gga.utc.get() - time_of_day_last) <
                    1e-3) {  // GGA messages for this dataset come second
                    ImagePoint::GPSData gps_data;
                    gps_data.seq = gps_utc_to_unix_time(date_last, gga.utc.get());
                    gps_data.latitude = gga.latitude.get();
                    gps_data.longitude = gga.longitude.get();
                    if (gga.altitude.exists()) gps_data.altitude = gga.altitude.get();
                    time_list_gps.push_back(std::make_pair(gps_data.seq, gps_data));
                }
            }
        } else if (nmea_sentence->type() == "RMC") {
            nmea::rmc rmc(*nmea_sentence);
            if (rmc.utc.exists() && rmc.date.exists()) {
                date_last = rmc.date.get();
                time_of_day_last = rmc.utc.get();
            }
        } else if (nmea_sentence->type() == "GST") {
            std::optional<GSTData> gst_data = parse_gpgst(nmea_sentence->nmea_string());
            if (gst_data && std::abs(gst_data->utc_time - time_of_day_last) <
                                1e-3) {  // GST message for this dataset come third
                size_t unix_time_ns = gps_utc_to_unix_time(date_last, gst_data->utc_time);
                if (time_list_gps.back().first == unix_time_ns) {
                    time_list_gps.back().second.uncertainty.emplace(
                        gst_data->sigma_lat, gst_data->sigma_lon, gst_data->sigma_alt,
                        gst_data->error_orientation, gst_data->rms_range_error);
                }
            }
        }
    }
    return time_list_gps;
}

std::vector<std::string> split_nmea_sentence(const std::string& sentence) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(sentence);

    while (std::getline(ss, field, ',')) {
        // remove checksum from the last field if present
        auto asterisk = field.find('*');
        if (asterisk != std::string::npos) field = field.substr(0, asterisk);
        fields.push_back(field);
    }

    return fields;
}
double time_of_day_seconds(const double utc_time_hhmmss) {
    int hours = static_cast<int>(utc_time_hhmmss / 10000);
    int minutes = static_cast<int>((utc_time_hhmmss - hours * 10000) / 100);
    double seconds = utc_time_hhmmss - hours * 10000 - minutes * 100;
    return hours * 3600 + minutes * 60 + seconds;
}
std::optional<GSTData> parse_gpgst(const std::string& sentence) {
    if (sentence.substr(0, 6) != "$GPGST") {
        return std::nullopt;
    }

    std::vector<std::string> fields = split_nmea_sentence(sentence);
    if (fields.size() != 9) {
        return std::nullopt;
    }
    for (size_t i = 0; i < fields.size(); i++) {
        if (fields[i].empty()) return std::nullopt;
    }

    GSTData gst;
    gst.utc_time = time_of_day_seconds(std::stod(fields[1]));
    gst.rms_range_error = std::stod(fields[2]);
    gst.error_semi_major = std::stod(fields[3]);
    gst.error_semi_minor = std::stod(fields[4]);
    gst.error_orientation = std::stod(fields[5]);
    gst.sigma_lat = std::stod(fields[6]);
    gst.sigma_lon = std::stod(fields[7]);
    gst.sigma_alt = std::stod(fields[8]);

    return gst;
}

}  // namespace gps_parser_help

template <typename T>
size_t abs_diff(const T& a, const T& b) {
    return a > b ? a - b : b - a;
}
}  // namespace detail
}  // namespace robot::experimental::learn_descriptors