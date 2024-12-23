#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

#include "common/check.hh"

namespace robot::experimental::learn_descriptors {
DataParser::DataParser(const std::filesystem::path &image_root_dir,
                       const std::vector<std::string> &survey_list) : 
                       image_root_dir_(image_root_dir) , survey_list_(survey_list) {
    ROBOT_CHECK(std::filesystem::exists(image_root_dir), "Image root dir does not exist!",
                image_root_dir);
    surveys_.load(image_root_dir.string(), survey_list);
}
DataParser::~DataParser() {}

Eigen::Affine3d DataParser::get_T_world_camera(size_t survey_idx, size_t img_idx, bool use_gps, bool use_compass) {
    const symphony_lake_dataset::ImagePoint img_point = surveys_.get(survey_idx).getImagePoint(img_idx);
    Eigen::Vector3d t;
    Eigen::Matrix3d R;        
    Eigen::Affine3d T_world_camera;
    if (use_gps) {
        t[0] = img_point.x;
        t[1] = img_point.y;
    }
    // coordinate frame is x-axis north, y-axis east, z-axis down (to Earth's core)
    double yaw = use_compass ? img_point.theta - img_point.pan : img_point.pan;    
    Eigen::Matrix3d R_y(Eigen::AngleAxisd(img_point.tilt, Eigen::Vector3d::UnitY()));
    Eigen::Matrix3d R_x = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_z(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    R = R_x * R_y * R_z;
    T_world_camera = t * R;
    return T_world_camera;
}
}  // namespace robot::experimental::learn_descriptors