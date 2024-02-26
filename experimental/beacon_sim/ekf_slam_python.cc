
#include "experimental/beacon_sim/ekf_slam.hh"
#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(ekf_slam_python, m) {
    m.doc() = "EKF Slam";

    py::class_<EkfSlamConfig>(m, "EkfSlamConfig")
        .def(py::init<int, double, double, double, double, double, double, double, double,
                      double, double, double>(),
             "max_num_beacons"_a, "initial_beacon_uncertainty_m"_a, "along_track_process_noise_m_per_rt_meter"_a,
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
}
}  // namespace robot::experimental::beacon_sim
