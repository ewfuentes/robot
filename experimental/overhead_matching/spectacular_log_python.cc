
#include <iostream>

#include "experimental/overhead_matching/spectacular_log.hh"
#include "opencv2/core/eigen.hpp"
#include "pybind11/eigen.h"
#include "pybind11/eigen/tensor.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl/filesystem.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::overhead_matching {
namespace {
Eigen::Tensor<uint8_t, 3> tensor_map_from_cv_mat(const cv::Mat &mat) {
    Eigen::Tensor<uint8_t, 3> tensor;
    cv::cv2eigen(mat, tensor);
    return tensor;
}
}  // namespace
PYBIND11_MODULE(spectacular_log_python, m) {
    m.doc() = "spectacular log";

    py::module_::import("common.time.robot_time_python");

    py::class_<FrameCalibration>(m, "FrameCalibration")
        .def(py::init<>())
        .def_readwrite("focal_length", &FrameCalibration::focal_length)
        .def_readwrite("principal_point", &FrameCalibration::principal_point)
        .def_readwrite("exposure_time", &FrameCalibration::exposure_time_s)
        .def_readwrite("depth_scale", &FrameCalibration::depth_scale);

    py::class_<FrameGroup>(m, "FrameGroup")
        .def(py::init<>())
        .def_readwrite("time_of_validity", &FrameGroup::time_of_validity)
        .def("bgr_frame",
             [](const FrameGroup &self) { return tensor_map_from_cv_mat(self.bgr_frame); })
        .def("depth_frame",
             [](const FrameGroup &self) { return tensor_map_from_cv_mat(self.depth_frame); })
        .def_readwrite("color_calibration", &FrameGroup::color_calibration)
        .def_readwrite("depth_calibration", &FrameGroup::depth_calibration);

    py::class_<ImuSample>(m, "ImuSample")
        .def(py::init<>())
        .def_readwrite("time_of_validity", &ImuSample::time_of_validity)
        .def_readwrite("accel_mpss", &ImuSample::accel_mpss)
        .def_readwrite("gyro_radps", &ImuSample::gyro_radps);

    py::class_<SpectacularLog>(m, "SpectacularLog")
        .def(py::init<std::filesystem::path>())
        .def("get_imu_sample", &SpectacularLog::get_imu_sample)
        .def("get_frame", &SpectacularLog::get_frame)
        .def("min_imu_time", &SpectacularLog::min_imu_time)
        .def("max_imu_time", &SpectacularLog::max_imu_time)
        .def("min_frame_time", &SpectacularLog::min_frame_time)
        .def("max_frame_time", &SpectacularLog::max_frame_time)
        .def("num_frames", &SpectacularLog::num_frames);
}
}  // namespace robot::experimental::overhead_matching
