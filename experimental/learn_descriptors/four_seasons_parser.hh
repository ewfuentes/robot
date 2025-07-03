#pragma once

#include <stdio.h>

#include <cstddef>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/image_point.hh"
#include "nmea/message/gga.hpp"
#include "nmea/message/rmc.hpp"
#include "nmea/sentence.hpp"

namespace robot::experimental::learn_descriptors {
class FourSeasonsParser {
   public:
    static constexpr double CAM_HZ = 30.0;
    static constexpr double CAM_CAP_DELTA = 1e9 / CAM_HZ;
    struct CameraCalibrationFisheye {
        double fx, fy, cx, cy, k1, k2, k3, k4;
    };
    FourSeasonsParser(const std::filesystem::path& root_dir,
                      const std::filesystem::path& calibration_dir);
    cv::Mat load_image(const size_t idx) const;
    const ImagePoint& get_image_point(const size_t idx) const { return img_pt_vector_[idx]; };
    size_t num_images() const { return img_pt_vector_.size(); };
    const CameraCalibrationFisheye& camera_calibration() const { return cal_; };
    const liegroups::SE3& S_from_AS() const {
        return transforms_.S_from_AS;
    };  // metric scale from arbitrary (internal slam) scale
    const liegroups::SE3& cam_from_imu() const { return transforms_.cam_from_imu; };
    const liegroups::SE3& w_from_gpsw() const {
        return transforms_.w_from_gpsw;
    };  // visual world from local gps (ENU)
    const liegroups::SE3& gps_from_imu() const {
        return transforms_.gps_from_imu;
    };  // phsyical onboard gps from physical onboard imu
    const liegroups::SE3& e_from_gpsw() const {
        return transforms_.e_from_gpsw;
    };  // ECEF from local gps (ENU)
    double gnss_scale() const {
        return transforms_.gnss_scale;
    };  // scale from vio frame to gnss frame. WARNING: will require retooling if the scales per
        // keyframe (pose) are not all one value. See more here:
        // https://github.com/pmwenzel/4seasons-dataset

    static Eigen::Vector3d gcs_from_ECEF(const Eigen::Vector3d& t_place_from_ECEF);
    static Eigen::Vector3d ECEF_from_gcs(const Eigen::Vector3d& gcs_coordinate);

   protected:
    struct FourSeasonsTransforms {
        liegroups::SE3 S_from_AS;
        liegroups::SE3 cam_from_imu;
        liegroups::SE3 w_from_gpsw;
        liegroups::SE3 gps_from_imu;
        liegroups::SE3 e_from_gpsw;
        double gnss_scale;

        FourSeasonsTransforms(const std::filesystem::path& path_transforms);

       private:
        static liegroups::SE3 get_transform_from_line(const std::string& line);
    };
    const std::filesystem::path root_dir_;
    const std::filesystem::path img_dir_;
    const CameraCalibrationFisheye cal_;
    const FourSeasonsTransforms transforms_;
    ImagePointVector img_pt_vector_;
};

namespace detail {
template <typename T>
std::size_t abs_diff(const T& a, const T& b);
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
FourSeasonsParser::CameraCalibrationFisheye load_camera_calibration(
    const std::filesystem::path& calibration_dir);
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
using TimeGPSList = std::vector<std::pair<size_t, ImagePoint::GPSData>>;
struct GSTData {
    double utc_time;
    double rms_range_error;
    double error_semi_major;
    double error_semi_minor;
    double error_orientation;
    double sigma_lat;
    double sigma_lon;
    double sigma_alt;
};
std::vector<std::string> split_nmea_sentence(const std::string& nmea_sentence);
double time_of_day_seconds(const double utc_time_hhmmss);
std::optional<GSTData> parse_gpgst(const std::string& nmea_sentence);
size_t gps_utc_to_unix_time(const nmea::date& utc_date, const double utc_time_day_seconds);
TimeGPSList create_gps_time_data_list(const std::filesystem::path& path_gps);
}  // namespace gps_parser_help
}  // namespace detail
}  // namespace robot::experimental::learn_descriptors
