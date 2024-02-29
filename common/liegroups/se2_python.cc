
#include "common/liegroups/se2.hh"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

#include <iostream>

namespace py = pybind11;

namespace robot::liegroups {

PYBIND11_MODULE(se2_python, m) {
    py::module_::import("common.liegroups.so2_python");
    py::class_<SE2>(m, "SE2")
        // Construct from rotation angle
        .def(py::init(&SE2::rot))
        // Construct from translation
        .def(py::init([](const Eigen::Vector2d &trans) { return SE2::trans(trans); }))
        .def(py::init<const double &, const Eigen::Vector2d &>())
        .def_static("exp", [](const Eigen::Vector3d &tangent) -> SE2 { return SE2::exp(tangent); })
        .def("so2", [](const SE2 &a) -> SO2 { return a.so2(); })
        .def("translation", py::overload_cast<>(&SE2::translation, py::const_))
        .def("translation", py::overload_cast<>(&SE2::translation))
        .def("log", &SE2::log)
        .def("matrix", &SE2::matrix)
        .def("inverse", [](const SE2 &a) -> SE2 { return a.inverse(); })
        .def(py::self * Eigen::Vector2d())
        .def(
            "__mul__", [](const SE2 &a, const SE2 &b) -> SE2 { return a * b; }, py::is_operator())
        .def(
            "__imul__",
            [](SE2 &a, const SE2 &b) -> SE2 & {
                a *= b;
                return a;
            },
            py::is_operator());
}
}  // namespace robot::liegroups
