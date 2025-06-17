#include "experimental/learn_descriptors/four_seasons_parser.hh"

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
#include "common/liegroups/se3.hh"
#include "gtest/gtest.h"

namespace lrn_descs = robot::experimental::learn_descriptors;

class FourSeasonsParserTestHelper {
   public:
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

    static std::vector<std::string> parse_line_adv(const std::string& line,
                                                   const std::string& delim = " ") {
        if (delim == " ") {
            return std::vector<std::string>(
                absl::StrSplit(line, absl::ByAnyChar(" \t\n\r"), absl::SkipWhitespace()));
        }
        return std::vector<std::string>(absl::StrSplit(line, delim, absl::SkipWhitespace()));
    }
    static bool images_equal(cv::Mat img1, cv::Mat img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            return false;
        }
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        diff = diff.reshape(1);
        return cv::countNonZero(diff) == 0;
    }

    template <typename T>
    static T round_to_sig_figs(T val, int n) {
        if (val == 0) return 0;
        double d = static_cast<double>(val);
        int exponent = static_cast<int>(std::floor(std::log10(std::abs(d))));
        double multiplier = std::pow(10.0, n - exponent - 1);
        return static_cast<T>(std::round(d * multiplier) / multiplier);
    }

    // minimum sig figs of time in seconds (of type double) for all entries of results.txt
    static size_t min_sig_figs_result_time(const std::filesystem::path& path_result) {
        std::ifstream file_result(path_result);
        std::string line;
        int min_figs = INT_MAX;
        while (std::getline(file_result, line)) {
            const std::vector<std::string> parsed_line = parse_line_adv(line, " ");
            const std::string str_time_seconds =
                parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)];
            const int figs =
                str_time_seconds.size() +
                (str_time_seconds.find('.') != std::string::npos ? -1 : 0);  // '.' is not a fig
            min_figs = std::min(min_figs, figs);
        }
        return min_figs;
    }

    using TimeDataMap = std::unordered_map<size_t, std::vector<std::string>>;
    using TimeDataList = std::vector<std::pair<size_t, std::vector<std::string>>>;

    static const TimeDataList create_img_time_data_list(const std::filesystem::path& path_img,
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

    static const TimeDataMap create_gps_time_data_map(const std::filesystem::path& path_gps,
                                                      const size_t time_sig_figs) {
        TimeDataMap time_map_gps;
        std::ifstream file_gps(path_gps);
        std::string line;
        std::getline(file_gps,
                     line);  // advance a line to get past the top comment in GNSSPoses.txt
        while (std::getline(file_gps, line)) {
            std::vector<std::string> parsed_line = parse_line_adv(line, ",");
            size_t time_ns = std::stoull(parsed_line[static_cast<size_t>(GPSIdx::TIME_NS)]);
            time_map_gps.insert({round_to_sig_figs(time_ns, time_sig_figs), parsed_line});
        }
        return time_map_gps;
    }

    static const TimeDataMap create_result_time_data_map(const std::filesystem::path& path_result,
                                                         const size_t time_sig_figs) {
        TimeDataMap time_map_result;
        std::ifstream file_result(path_result);
        std::string line;
        while (std::getline(file_result, line)) {
            std::vector<std::string> parsed_line = parse_line_adv(line, " ");
            double time_seconds = std::stod(parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)]);
            size_t time_ns = time_seconds * 1e9;
            time_map_result.insert({round_to_sig_figs(time_ns, time_sig_figs), parsed_line});
        }
        return time_map_result;
    }

    static lrn_descs::FourSeasonsParser::CameraCalibrationFisheye load_camera_calibration(
        const std::filesystem::path& calibration_dir) {
        std::ifstream file_calibration(calibration_dir / "calib_0.txt");
        std::string line_calibration;
        std::getline(file_calibration, line_calibration);
        std::vector<std::string> parsed_calib_line = parse_line_adv(line_calibration, " ");
        return lrn_descs::FourSeasonsParser::CameraCalibrationFisheye{
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FX)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::FY)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CX)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::CY)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K1)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K2)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K3)]),
            std::stod(parsed_calib_line[static_cast<size_t>(CalibIdx::K4)])};
    }
};

using Helper = FourSeasonsParserTestHelper;

namespace robot::experimental::learn_descriptors {
TEST(FourSeasonsParserTest, parser_test) {
    const std::filesystem::path dir_snippet =
        "external/four_seasons_snippet/recording_2020-04-07_11-33-45";
    const std::filesystem::path dir_calibration = "external/four_seasons_snippet/calibration";

    EXPECT_TRUE(std::filesystem::exists(dir_snippet));
    EXPECT_TRUE(std::filesystem::exists(dir_calibration));

    FourSeasonsParser parser(dir_snippet, dir_calibration);

    // transformations test
    const liegroups::SE3 AS_from_S(Eigen::Quaterniond(0.999992, 0.000869, 0.003288, -0.002016),
                                   Eigen::Vector3d::Zero());
    const liegroups::SE3 imu_from_cam(Eigen::Quaterniond(-0.002350, -0.007202, 0.708623, -0.705546),
                                      Eigen::Vector3d(0.175412, 0.003689, -0.058106));
    const liegroups::SE3 gpsw_from_w(Eigen::Quaterniond(0.802501, 0.000344, -0.000541, 0.596650),
                                     Eigen::Vector3d(0.256671, 0.021142, 0.073751));
    const liegroups::SE3 imu_from_gps(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
    const liegroups::SE3 gpsw_from_e(
        Eigen::Quaterniond(0.590438, 0.224931, 0.275937, 0.724326),
        Eigen::Vector3d(4164702.580389, 857109.771387, 4738828.771006));
    const double gnss_scale = 0.911501;
    EXPECT_TRUE(AS_from_S.matrix().isApprox(parser.get_AS_from_S().matrix()));
    EXPECT_TRUE(imu_from_cam.matrix().isApprox(parser.get_imu_from_cam().matrix()));
    EXPECT_TRUE(gpsw_from_w.matrix().isApprox(parser.get_gpsw_from_w().matrix()));
    EXPECT_TRUE(imu_from_gps.matrix().isApprox(parser.get_imu_from_gps().matrix()));
    EXPECT_TRUE(gpsw_from_e.matrix().isApprox(parser.get_gpsw_from_e().matrix()));
    EXPECT_DOUBLE_EQ(gnss_scale, parser.get_gnss_scale());

    // calibration test
    const FourSeasonsParser::CameraCalibrationFisheye calibration_target =
        FourSeasonsParserTestHelper::load_camera_calibration(dir_calibration);
    const FourSeasonsParser::CameraCalibrationFisheye calibration = parser.get_camera_calibration();
    EXPECT_DOUBLE_EQ(calibration_target.cx, calibration.cx);
    EXPECT_DOUBLE_EQ(calibration_target.cy, calibration.cy);
    EXPECT_DOUBLE_EQ(calibration_target.fx, calibration.fx);
    EXPECT_DOUBLE_EQ(calibration_target.fy, calibration.fy);
    EXPECT_DOUBLE_EQ(calibration_target.k1, calibration.k1);
    EXPECT_DOUBLE_EQ(calibration_target.k2, calibration.k2);
    EXPECT_DOUBLE_EQ(calibration_target.k3, calibration.k3);
    EXPECT_DOUBLE_EQ(calibration_target.k4, calibration.k4);

    EXPECT_NE(parser.num_images(), 0);

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);

    const std::filesystem::path dir_img = dir_snippet / "distorted_images/cam0";
    const std::filesystem::path path_img_time = dir_snippet / "times.txt";
    const std::filesystem::path path_gps = dir_snippet / "GNSSPoses.txt";
    const std::filesystem::path path_result = dir_snippet / "result.txt";
    const size_t min_sig_figs_time = Helper::min_sig_figs_result_time(path_result);

    // fetch data for all the fields (gps, img times, result)
    const Helper::TimeDataList img_time_list =
        Helper::create_img_time_data_list(path_img_time, min_sig_figs_time);
    EXPECT_EQ(img_time_list.size(), parser.num_images());
    const Helper::TimeDataMap gps_time_map =
        Helper::create_gps_time_data_map(path_gps, min_sig_figs_time);
    const Helper::TimeDataMap result_time_map =
        Helper::create_result_time_data_map(path_result, min_sig_figs_time);
    for (size_t i = 0; i < parser.num_images(); i++) {
        const ImagePoint& img_pt = parser.get_image_point(i);
        const cv::Mat img = parser.load_image(i);

        // seq (time in nanoseconds) test
        EXPECT_EQ(std::stod(img_time_list[i].second[0]), img_pt.seq);

        // image testimg_time_list[i].first
        const std::filesystem::path path_img = dir_img / (std::to_string(img_pt.seq) + ".png");
        const cv::Mat img_target = cv::imread(path_img);
        EXPECT_TRUE(Helper::images_equal(img_target, img));

        // check gps and result entries
        if (gps_time_map.find(img_time_list[i].first) != gps_time_map.end()) {
            const std::vector<std::string>& parsed_line_gps =
                gps_time_map.at(img_time_list[i].first);

            Eigen::Vector3d gps_translation(
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::TRAN_X)]),
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::TRAN_Y)]),
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::TRAN_Z)]));
            Eigen::Quaterniond gps_quat(
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::QUAT_W)]),
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::QUAT_X)]),
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::QUAT_Y)]),
                std::stod(parsed_line_gps[static_cast<size_t>(Helper::GPSIdx::QUAT_Z)]));
            EXPECT_TRUE(
                liegroups::SE3(gps_quat, gps_translation).matrix().isApprox(img_pt.gps->matrix()));
        } else if (img_pt.gps.has_value()) {
            throw std::runtime_error("img_pt has gps point but files do not!");
        }
        if (result_time_map.find(img_time_list[i].first) != result_time_map.end()) {
            const std::vector<std::string>& parsed_line_ground_truth =
                result_time_map.at(img_time_list[i].first);
            Eigen::Vector3d ground_truth_translation(
                std::stod(parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::TRAN_X)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::TRAN_Y)]),
                std::stod(
                    parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::TRAN_Z)]));
            Eigen::Quaterniond ground_truth_quat(
                std::stod(parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::QUAT_W)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::QUAT_X)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::QUAT_Y)]),
                std::stod(
                    parsed_line_ground_truth[static_cast<size_t>(Helper::ResultIdx::QUAT_Z)]));
            EXPECT_TRUE(liegroups::SE3(ground_truth_quat, ground_truth_translation)
                            .matrix()
                            .isApprox(img_pt.ground_truth->matrix()));
        } else if (img_pt.ground_truth.has_value()) {
            throw std::runtime_error("img_pt has ground_truth point but files do not!");
        }
    }
}
}  // namespace robot::experimental::learn_descriptors