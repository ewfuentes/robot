#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "experimental/learn_descriptors/camera_calibration.hh"
#include "experimental/learn_descriptors/gps_data.hh"
#include "nmea/object/date.hpp"

namespace robot::experimental::learn_descriptors::detail::four_seasons_parser {
template <typename T>
std::size_t abs_diff(const T& a, const T& b) {
    return a > b ? a - b : b - a;
};
namespace txt_parser_help {
using TimeDataMap = std::unordered_map<size_t, std::vector<std::string>>;
using TimeDataList = std::vector<std::pair<size_t, std::vector<std::string>>>;
enum class GPSIdx {
    TIME_NS = 0,
    TRAN_X = 1,
    TRAN_Y = 2,
    TRAN_Z = 3,
    QUAT_X = 4,
    QUAT_Y = 5,
    QUAT_Z = 6,
    QUAT_W = 7,
};
enum class ImgIdx {
    TIME_NS = 0,
    TIME_SEC = 1,
    EXPOSURE_TIME = 2  // I'm not completely sure if this is correct - Nico
};
enum class ResultIdx {
    TIME_SEC = 0,
    TRAN_X = 1,
    TRAN_Y = 2,
    TRAN_Z = 3,
    QUAT_X = 4,
    QUAT_Y = 5,
    QUAT_Z = 6,
    QUAT_W = 7,
};
enum class CalibIdx { FX = 1, FY = 2, CX = 3, CY = 4, K1 = 5, K2 = 6, K3 = 7, K4 = 8 };
std::vector<std::string> parse_line_adv(const std::string& line, const std::string& delim = " ");
CameraCalibrationFisheye load_camera_calibration(const std::filesystem::path& calibration_dir);
template <typename T>
T round_to_sig_figs(T val, int n) {
    if (val == 0) return 0;
    double d = static_cast<double>(val);
    int exponent = static_cast<int>(std::floor(std::log10(std::abs(d))));
    double multiplier = std::pow(10.0, n - exponent - 1);
    return static_cast<T>(std::round(d * multiplier) / multiplier);
}
size_t min_sig_figs_result_time(const std::filesystem::path& path_vio);
const TimeDataList create_img_time_data_list(const std::filesystem::path& path_img,
                                             const size_t time_sig_figs);
const TimeDataMap create_gnss_poses_time_data_map(const std::filesystem::path& path_gnss,
                                                  const size_t time_sig_figs);
const TimeDataMap create_vio_time_data_map(const std::filesystem::path& path_vio,
                                           const size_t time_sig_figs);
}  // namespace txt_parser_help
namespace gps_parser_help {
using TimeGPSList = std::vector<std::pair<size_t, GPSData>>;
struct GSTData {
    double utc_time_ns;
    double rms_range_error_m;
    double error_semi_major_m;
    double error_semi_minor_m;
    double error_orientation_deg;
    double sigma_lat_m;
    double sigma_lon_m;
    double sigma_alt_m;
};
std::vector<std::string> split_nmea_sentence(const std::string& nmea_sentence);
double time_of_day_seconds(const double utc_time_hhmmss);
std::optional<GSTData> parse_gpgst(const std::string& nmea_sentence);
size_t gps_utc_to_unix_time(const nmea::date& utc_date, const double utc_time_day_seconds);
TimeGPSList create_gps_time_data_list(const std::filesystem::path& path_gps);
}  // namespace gps_parser_help
}  // namespace robot::experimental::learn_descriptors::detail::four_seasons_parser
