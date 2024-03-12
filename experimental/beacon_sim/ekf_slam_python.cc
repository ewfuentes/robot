
#include <filesystem>

#include "common/proto/load_from_file.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(ekf_slam_python, m) {
    m.doc() = "EKF Slam";

    py::class_<EkfSlamConfig>(m, "EkfSlamConfig")
        .def(py::init<int, double, double, double, double, double, double, double, double, double,
                      double, double>(),
             "max_num_beacons"_a, "initial_beacon_uncertainty_m"_a,
             "along_track_process_noise_m_per_rt_meter"_a,
             "cross_track_process_noise_m_per_rt_meter"_a, "pos_process_noise_m_per_rt_s"_a,
             "heading_process_noise_rad_per_rt_meter"_a, "heading_process_noise_rad_per_rt_s"_a,
             "beacon_pos_process_noise_m_per_rt_s"_a, "range_measurement_noise_m"_a,
             "bearing_measurement_noise_rad"_a, "on_map_load_position_uncertainty_m"_a,
             "on_map_load_heading_uncertainty_rad"_a)
        .def_readwrite("max_num_beacons", &EkfSlamConfig::max_num_beacons)
        .def_readwrite("initial_beacon_uncertainty_m", &EkfSlamConfig::initial_beacon_uncertainty_m)
        .def_readwrite("along_track_process_noise_m_per_rt_meter",
                       &EkfSlamConfig::along_track_process_noise_m_per_rt_meter)
        .def_readwrite("cross_track_process_noise_m_per_rt_meter",
                       &EkfSlamConfig::cross_track_process_noise_m_per_rt_meter)
        .def_readwrite("pos_process_noise_m_per_rt_s", &EkfSlamConfig::pos_process_noise_m_per_rt_s)
        .def_readwrite("heading_process_noise_rad_per_rt_meter",
                       &EkfSlamConfig::heading_process_noise_rad_per_rt_meter)
        .def_readwrite("heading_process_noise_rad_per_rt_s",
                       &EkfSlamConfig::heading_process_noise_rad_per_rt_s)
        .def_readwrite("beacon_pos_process_noise_m_per_rt_s",
                       &EkfSlamConfig::beacon_pos_process_noise_m_per_rt_s)
        .def_readwrite("range_measurement_noise_m", &EkfSlamConfig::range_measurement_noise_m)
        .def_readwrite("bearing_measurement_noise_rad",
                       &EkfSlamConfig::bearing_measurement_noise_rad)
        .def_readwrite("on_map_load_position_uncertainty_m",
                       &EkfSlamConfig::on_map_load_position_uncertainty_m)
        .def_readwrite("on_map_load_heading_uncertainty_rad",
                       &EkfSlamConfig::on_map_load_heading_uncertainty_rad);

    py::class_<EkfSlamEstimate>(m, "EkfSlamEstimate")
        .def_readwrite("time_of_validity", &EkfSlamEstimate::time_of_validity)
        .def_readwrite("mean", &EkfSlamEstimate::mean)
        .def_readwrite("cov", &EkfSlamEstimate::cov)
        .def_readwrite("beacon_ids", &EkfSlamEstimate::beacon_ids)
        .def("local_from_robot",
             py::overload_cast<const liegroups::SE2 &>(&EkfSlamEstimate::local_from_robot))
        .def("local_from_robot",
             py::overload_cast<>(&EkfSlamEstimate::local_from_robot, py::const_))
        .def("robot_cov", &EkfSlamEstimate::robot_cov)
        .def("beacon_in_local", &EkfSlamEstimate::beacon_in_local)
        .def("beacon_cov", &EkfSlamEstimate::beacon_cov);

    py::class_<EkfSlam>(m, "EkfSlam")
        .def(py::init<const EkfSlamConfig &, const time::RobotTimestamp &>())
        .def(py::init<const EkfSlam &>())
        .def("load_map",
             [](EkfSlam &ekf, const std::string &path) -> bool {
                 const std::filesystem::path load_path = path;
                 const auto maybe_mapped_landmarks =
                     robot::proto::load_from_file<proto::MappedLandmarks>(load_path);
                 if (!maybe_mapped_landmarks.has_value()) {
                     return false;
                 }
                 const auto mapped_landmarks = unpack_from(maybe_mapped_landmarks.value());
                 constexpr bool LOAD_OFF_DIAGONALS = true;
                 ekf.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);
                 return true;
             })
        .def("save_map",
             [](const EkfSlam &ekf, const std::string &path) -> bool {
                 const std::filesystem::path save_path = path;
                 const auto mapped_landmarks = extract_mapped_landmarks(ekf.estimate());

                 std::filesystem::create_directories(save_path.parent_path());
                 std::ofstream out(save_path);
                 if (!out.is_open()) {
                     return false;
                 }

                 proto::MappedLandmarks proto;
                 pack_into(mapped_landmarks, &proto);
                 proto.SerializeToOstream(&out);
                 return true;
             })
        .def("predict", &EkfSlam::predict)
        .def("update", &EkfSlam::update)
        .def_property("estimate", py::overload_cast<>(&EkfSlam::estimate, py::const_),
                      py::overload_cast<>(&EkfSlam::estimate))
        .def("config", &EkfSlam::config);
}
}  // namespace robot::experimental::beacon_sim
