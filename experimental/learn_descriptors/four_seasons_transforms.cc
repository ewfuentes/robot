#include "experimental/learn_descriptors/four_seasons_transforms.hh"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/check.hh"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/four_seasons_parser_detail.hh"

namespace robot::experimental::learn_descriptors {
FourSeasonsTransforms::StaticTransforms::StaticTransforms(
    const std::filesystem::path& path_transforms) {
    std::ifstream file_transforms(path_transforms);
    ROBOT_CHECK(file_transforms, path_transforms);
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

liegroups::SE3 FourSeasonsTransforms::StaticTransforms::get_transform_from_line(
    const std::string& line) {
    enum TransformEntry { T_X, T_Y, T_Z, Q_X, Q_Y, Q_Z, Q_W };
    std::vector<std::string> parsed_transform_line =
        detail::four_seasons_parser::txt_parser_help::parse_line_adv(line, ",");
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
}  // namespace robot::experimental::learn_descriptors