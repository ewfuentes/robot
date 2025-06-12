#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <limits.h>

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>

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

namespace lrn_descs = robot::experimental::learn_descriptors;

std::vector<std::string> parse_line_adv(const std::string& line, const std::string& delim = " ") {
    std::vector<std::string> parsedline;
    if (delim != " ") {
        size_t start = 0;
        size_t end;
        while ((end = line.find(delim, start)) != std::string::npos) {
            parsedline.push_back(line.substr(start, end - start));
            start = end + delim.length();
        }
        parsedline.push_back(line.substr(start));
        return parsedline;
    }

    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace(line[i])) i++;

        if (i == line.size()) break;

        size_t start = i;
        while (i < line.size() && !std::isspace(line[i])) i++;

        parsedline.push_back(line.substr(start, i - start));
    }
    return parsedline;
}

lrn_descs::FourSeasonsParser::CameraCalibrationFisheye load_camera_calibration(
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

template <typename T>
T round_to_sig_figures(T val, int n) {
    if (val == 0) return 0;
    double d = static_cast<double>(val);
    int exponent = static_cast<int>(std::floor(std::log10(std::abs(d))));
    double multiplier = std::pow(10.0, n - exponent - 1);
    return static_cast<T>(std::round(d * multiplier) / multiplier);
}

// minimum sig figs of time in seconds (of type double) for all entries of results.txt
size_t min_sig_figs_result_time(const std::filesystem::path& path_result) {
    std::ifstream file_result(path_result);
    std::string line;
    int min_figs = INT_MAX;
    while (std::getline(file_result, line)) {
        const std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        const std::string str_time_seconds = parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)];
        const int figs =
            str_time_seconds.size() +
            (str_time_seconds.find('.') != std::string::npos ? -1 : 0);  // '.' is not a fig
        min_figs = std::min(min_figs, figs);
    }
    return min_figs;
}

typedef std::unordered_map<size_t, std::vector<std::string>> TimeDataMap;
typedef std::vector<std::pair<size_t, std::vector<std::string>>> TimeDataList;

const TimeDataList create_img_time_data_list(const std::filesystem::path& path_img,
                                             const size_t time_sig_figs) {
    TimeDataList time_map_img;
    std::ifstream file_img(path_img);
    std::string line;
    while (std::getline(file_img, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        size_t time_ns = std::stoul(parsed_line[static_cast<size_t>(ImgIdx::TIME_NS)]);
        time_map_img.push_back(
            std::make_pair(round_to_sig_figures(time_ns, time_sig_figs), parsed_line));
    }
    return time_map_img;
}

const TimeDataMap create_gps_time_data_map(const std::filesystem::path& path_gps,
                                           const size_t time_sig_figs) {
    TimeDataMap time_map_gps;
    std::ifstream file_gps(path_gps);
    std::string line;
    std::getline(file_gps, line);  // advance a line to get past the top comment in GNSSPoses.txt
    while (std::getline(file_gps, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, ",");
        size_t time_ns = std::stoul(parsed_line[static_cast<size_t>(GPSIdx::TIME_NS)]);
        time_map_gps.insert({round_to_sig_figures(time_ns, time_sig_figs), parsed_line});
    }
    return time_map_gps;
}

const TimeDataMap create_result_time_data_map(const std::filesystem::path& path_result,
                                              const size_t time_sig_figs) {
    TimeDataMap time_map_result;
    std::ifstream file_result(path_result);
    std::string line;
    while (std::getline(file_result, line)) {
        std::vector<std::string> parsed_line = parse_line_adv(line, " ");
        double time_seconds = std::stod(parsed_line[static_cast<size_t>(ResultIdx::TIME_SEC)]);
        size_t time_ns = time_seconds * 1e9;
        time_map_result.insert({round_to_sig_figures(time_ns, time_sig_figs), parsed_line});
    }
    return time_map_result;
}

namespace robot::experimental::learn_descriptors {

FourSeasonsParser::FourSeasonsParser(const std::filesystem::path& root_dir,
                                     const std::filesystem::path& calibration_dir)
    : root_dir_(root_dir),
      img_dir_(root_dir / "distorted_images" / "cam0"),
      cal_(load_camera_calibration(calibration_dir)) {
    std::cout << "oooga" << std::endl;
    const std::filesystem::path path_img = root_dir_ / "times.txt";
    const std::filesystem::path path_gps = root_dir_ / "GNSSPoses.txt";
    const std::filesystem::path path_result = root_dir_ / "result.txt";
    size_t id = 0;

    const size_t min_time_sig_figs = min_sig_figs_result_time(path_result);
    TimeDataList img_time_list = create_img_time_data_list(path_img, min_time_sig_figs);
    TimeDataMap gps_time_map = create_gps_time_data_map(path_gps, min_time_sig_figs);
    TimeDataMap result_time_map = create_result_time_data_map(path_result, min_time_sig_figs);

    for (const std::pair<size_t, std::vector<std::string>>& pair_time_data : img_time_list) {
        const size_t time_key = pair_time_data.first;
        ImagePoint img_pt;
        img_pt.id = id;
        img_pt.seq = std::stoul(pair_time_data.second[static_cast<size_t>(ImgIdx::TIME_NS)]);
        if (gps_time_map.find(time_key) != gps_time_map.end()) {
            const std::vector<std::string>& parsed_line_gps = gps_time_map.at(time_key);

            img_pt.gps.translation() =
                Eigen::Vector3d(std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::TRAN_X)]),
                                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::TRAN_Y)]),
                                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::TRAN_Z)]));
            Eigen::Quaterniond gps_quat(
                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::QUAT_W)]),
                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::QUAT_X)]),
                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::QUAT_Y)]),
                std::stod(parsed_line_gps[static_cast<size_t>(GPSIdx::QUAT_Z)]));
            img_pt.gps.linear() = gps_quat.toRotationMatrix();
            img_pt.has_gps = true;
        } else {
            img_pt.has_gps = false;
            std::clog << "There is no gps data at img_pt with id: " << id << std::endl;
        }
        if (result_time_map.find(time_key) != result_time_map.end()) {
            const std::vector<std::string>& parsed_line_ground_truth = result_time_map.at(time_key);
            img_pt.ground_truth.translation() = Eigen::Vector3d(
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::TRAN_X)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::TRAN_Y)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::TRAN_Z)]));
            Eigen::Quaterniond ground_truth_quat(
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::QUAT_W)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::QUAT_X)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::QUAT_Y)]),
                std::stod(parsed_line_ground_truth[static_cast<size_t>(ResultIdx::QUAT_Z)]));
            img_pt.ground_truth.linear() = ground_truth_quat.toRotationMatrix();
            img_pt.has_ground_truth = true;
        } else {
            img_pt.has_ground_truth = false;
            std::clog << "There is no ground truth data at img_pt with id: " << id << std::endl;
        }
        img_pt_vector_.push_back(img_pt);
        id++;
    }
}

cv::Mat FourSeasonsParser::load_image(const size_t idx) {
    return img_pt_vector_[idx].load_image(img_dir_);
}

const ImagePoint& FourSeasonsParser::get_image_point(const size_t idx) {
    return img_pt_vector_[idx];
}
}  // namespace robot::experimental::learn_descriptors