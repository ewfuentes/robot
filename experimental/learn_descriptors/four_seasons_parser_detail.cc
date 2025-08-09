#include "experimental/learn_descriptors/four_seasons_parser_detail.hh"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "nmea/message/gga.hpp"
#include "nmea/message/rmc.hpp"
#include "nmea/sentence.hpp"

namespace robot::experimental::learn_descriptors::detail::four_seasons_parser {
namespace txt_parser_help {

std::vector<std::string> parse_line_adv(const std::string& line, const std::string& delim) {
    if (delim == " ") {
        return std::vector<std::string>(
            absl::StrSplit(line, absl::ByAnyChar(" \t\n\r"), absl::SkipWhitespace()));
    }
    return std::vector<std::string>(absl::StrSplit(line, delim, absl::SkipWhitespace()));
}

std::shared_ptr<CameraCalibrationFisheye> load_camera_calibration(
    const std::filesystem::path& calibration_dir) {
    std::ifstream file_calibration(calibration_dir / "calib_0.txt");
    std::string line_calibration;
    std::getline(file_calibration, line_calibration);
    std::vector<std::string> parsed_calib_line = parse_line_adv(line_calibration, " ");
    return std::make_shared<CameraCalibrationFisheye>(
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FX)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FY)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CX)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CY)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K1)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K2)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K3)]),
        std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K4)]));
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
                    GPSData gps_data;
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
            if (gst_data && std::abs(gst_data->utc_time_ns - time_of_day_last) <
                                1e-3) {  // GST message for this dataset come third
                size_t unix_time_ns = gps_utc_to_unix_time(date_last, gst_data->utc_time_ns);
                if (time_list_gps.back().first == unix_time_ns) {
                    time_list_gps.back().second.uncertainty.emplace(
                        gst_data->sigma_lat_m, gst_data->sigma_lon_m, gst_data->sigma_alt_m,
                        gst_data->error_orientation_deg, gst_data->rms_range_error_m);
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
    gst.utc_time_ns = time_of_day_seconds(std::stod(fields[1]));
    gst.rms_range_error_m = std::stod(fields[2]);
    gst.error_semi_major_m = std::stod(fields[3]);
    gst.error_semi_minor_m = std::stod(fields[4]);
    gst.error_orientation_deg = std::stod(fields[5]);
    gst.sigma_lat_m = std::stod(fields[6]);
    gst.sigma_lon_m = std::stod(fields[7]);
    gst.sigma_alt_m = std::stod(fields[8]);

    return gst;
}
}  // namespace gps_parser_help
}  // namespace robot::experimental::learn_descriptors::detail::four_seasons_parser