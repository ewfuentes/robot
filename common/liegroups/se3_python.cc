
#include <sstream>

#include "common/liegroups/so3.hh"
#include "common/liegroups/se3.hh"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::liegroups {

PYBIND11_MODULE(se3_python, m) {
    py::module_::import("common.liegroups.so3_python");
    py::class_<SE3>(m, "SE3")
        .def(py::init<>())
        // Construct from SO3
        .def(py::init<const SO3 &, const Eigen::Vector3d>())
        // Construct from translation
        .def(py::init([](const Eigen::Vector3d &trans) { return SE3::trans(trans); }))
        .def_static("exp", [](const SE3::Tangent &tangent) -> SE3 { return SE3::exp(tangent); })
        .def("so3", [](const SE3 &a) -> SO3 { return a.so3(); })
        .def("translation", py::overload_cast<>(&SE3::translation, py::const_))
        .def("translation", py::overload_cast<>(&SE3::translation))
        .def("log", &SE3::log)
        .def("matrix", &SE3::matrix)
        .def("inverse", [](const SE3 &a) -> SE3 { return a.inverse(); })
        .def(py::self * Eigen::Vector3d())
        .def(
            "__mul__", [](const SE3 &a, const SE3 &b) -> SE3 { return a * b; }, py::is_operator())
        .def(
            "__matmul__",
            [](const SE3 &a, const Eigen::Matrix3Xd &b) -> Eigen::Matrix3Xd {
                Eigen::Matrix3Xd out(3, b.cols());
                for (int i = 0; i < b.cols(); i++) {
                    out.col(i) = a * b.col(i);
                }
                return out;
            },
            py::is_operator())
        .def(
            "__imul__",
            [](SE3 &a, const SE3 &b) -> SE3 & {
                a *= b;
                return a;
            },
            py::is_operator())
        .def("__repr__", [](const SE3 &a) {
            const Eigen::IOFormat fmt(8);
            std::ostringstream ss;
            ss << a.matrix().format(fmt);
            return ss.str();
        });
}
}  // namespace robot::liegroups

