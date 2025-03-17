#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

#include "common/check.hh"

namespace robot::experimental::learn_descriptors {
const Eigen::Vector3d DataParser::t_boat_cam(-0.420, 0, 0.255);

const Eigen::Isometry3d DataParser::T_boat_gps((Eigen::Matrix4d() <<
    1, 0, 0, 0.080,
    0, 1, 0, -0.118,
    0, 0, 1, 0.010,
    0, 0, 0, 1
).finished());

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

Eigen::Affine3d DataParser::get_T_world_camera(size_t survey_idx, size_t img_idx, bool use_gps,
                                               bool use_compass) {
    const symphony_lake_dataset::ImagePoint img_point =
        surveys_.get(survey_idx).getImagePoint(img_idx);
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
    T_world_camera.translate(t);
    T_world_camera.rotate(R);
    return T_world_camera;
}

const Eigen::Isometry3d DataParser::get_T_boat_camera(const symphony_lake_dataset::ImagePoint &img_pt) {
    return get_T_boat_camera(img_pt.pan, img_pt.tilt);
}
const Eigen::Isometry3d DataParser::get_T_boat_camera(double theta_pan, double theta_tilt) {
    Eigen::AngleAxisd R_z(-theta_pan, Eigen::Vector3d::UnitZ()); // pan
    Eigen::AngleAxisd R_y(-theta_tilt, Eigen::Vector3d::UnitY()); // tilt
    Eigen::Matrix3d R_boat_gimbal(R_z * R_y); 
        
    Eigen::Matrix3d R_gimbal_camera = (Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY())).matrix();
    Eigen::Matrix3d R_boat_camera(R_boat_gimbal * R_gimbal_camera);
    
    Eigen::Isometry3d T_boat_cam;
    T_boat_cam.translation() = t_boat_cam;
    T_boat_cam.linear() = R_boat_camera;
    return T_boat_cam;
}

const Eigen::Matrix3d DataParser::get_R_world_boat(double theta_compass) {
    Eigen::Matrix3d R_world_imu = Eigen::AngleAxisd(theta_compass, Eigen::Vector3d::UnitZ()).matrix();
    Eigen::Matrix3d R_imu_boat = T_boat_imu.inverse().linear();
    Eigen::Matrix3d R_world_boat = R_world_imu * R_imu_boat;
    return R_world_boat;
}

const Eigen::Isometry3d DataParser::get_T_world_boat(const symphony_lake_dataset::ImagePoint &img_pt) {
    Eigen::Matrix3d R_world_boat = get_R_world_boat(img_pt.theta);
    
    Eigen::Isometry3d T_world_gps;
    T_world_gps.linear() =  R_world_boat * T_boat_gps.linear();
    T_world_gps.translation() = Eigen::Vector3d(img_pt.x, img_pt.y, 0);

    Eigen::Isometry3d T_world_boat = T_world_gps * T_boat_gps.inverse();
    return T_world_boat;
}
}  // namespace robot::experimental::learn_descriptors