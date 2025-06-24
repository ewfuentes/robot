#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

#include "common/check.hh"

namespace robot::experimental::learn_descriptors {
const Eigen::Vector3d DataParser::t_boat_cam(-0.430, 0, 0.255);

const Eigen::Isometry3d DataParser::T_boat_gps(
    (Eigen::Matrix4d() << 1, 0, 0, 0.080, 0, 1, 0, -0.1183, 0, 0, 1, 0.010, 0, 0, 0, 1).finished());

const Eigen::Isometry3d DataParser::T_boat_imu = []() {
    Eigen::Isometry3d T_boat_imu;
    T_boat_imu.translation() = Eigen::Vector3d(-0.494, 0, -0.089);
    T_boat_imu.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).matrix();
    return T_boat_imu;
}();

DataParser::DataParser(const std::filesystem::path &image_root_dir,
                       const std::vector<std::string> &survey_list)
    : image_root_dir_(image_root_dir), survey_list_(survey_list) {
    ROBOT_CHECK(std::filesystem::exists(image_root_dir), "Image root dir does not exist!",
                image_root_dir);
    surveys_.load(image_root_dir.string(), survey_list);
}
DataParser::~DataParser() {}

const Eigen::Isometry3d DataParser::get_boat_from_camera(
    const symphony_lake_dataset::ImagePoint &img_pt) {
    return get_boat_from_camera(img_pt.pan, img_pt.tilt);
}
const Eigen::Isometry3d DataParser::get_boat_from_camera(double theta_pan, double theta_tilt) {
    Eigen::AngleAxisd R_z(-theta_pan, Eigen::Vector3d::UnitZ());   // pan
    Eigen::AngleAxisd R_y(-theta_tilt, Eigen::Vector3d::UnitY());  // tilt
    Eigen::Matrix3d R_boat_gimbal(R_z * R_y);

    Eigen::Matrix3d R_gimbal_camera = (Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX()) *
                                       Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()))
                                          .matrix();
    Eigen::Matrix3d R_boat_camera(R_boat_gimbal * R_gimbal_camera);

    Eigen::Isometry3d T_boat_cam;
    T_boat_cam.translation() = t_boat_cam;
    T_boat_cam.linear() = R_boat_camera;
    return T_boat_cam;
}

const Eigen::Isometry3d DataParser::get_world_from_gps(
    const symphony_lake_dataset::ImagePoint &img_pt) {
    Eigen::Isometry3d world_from_gps;
    // img_pt.theta is in the North East Down frame, not lattitude longitude coords. However,
    // because of the alignment of the imu and gps, this rotation math holds.
    world_from_gps.linear() = Eigen::AngleAxisd(img_pt.theta, Eigen::Vector3d::UnitZ()).matrix();
    world_from_gps.translation() = Eigen::Vector3d(img_pt.x, img_pt.y, 0);
    return world_from_gps;
}

const Eigen::Isometry3d DataParser::get_world_from_boat(
    const symphony_lake_dataset::ImagePoint &img_pt) {
    // Eigen::Isometry3d world_from_gps;

    // Eigen::Matrix3d R_world_imu = get_R_world_imu(img_pt.theta);
    // Eigen::Isometry3d T_imu_gps = T_boat_imu.inverse() * T_boat_gps;
    // Eigen::Matrix3d R_world_gps = R_world_imu * T_imu_gps.linear();

    // world_from_gps.linear() = R_world_gps;
    // world_from_gps.translation() = Eigen::Vector3d(img_pt.x, img_pt.y, 0);

    // Eigen::Isometry3d world_from_boat = world_from_gps * T_boat_gps.inverse();

    Eigen::Isometry3d world_from_gps = get_world_from_gps(img_pt);
    Eigen::Isometry3d world_from_boat = world_from_gps * T_boat_gps.inverse();
    return world_from_boat;
}
}  // namespace robot::experimental::learn_descriptors