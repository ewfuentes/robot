#include "common/liegroups/so2.hh"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::liegroups {

PYBIND11_MODULE(so2_python, m) {
    py::class_<SO2>(m, "SO2")
        // Construct from rotation angle
        .def(py::init<double>())
        .def_static("exp", [](const double theta) -> SO2 { return SO2::exp(theta); })
        .def("log", &SO2::log)
        .def("inverse", [](const SO2 &a) -> SO2 { return a.inverse(); })
        .def("matrix", &SO2::matrix)
        .def(
            "__mul__", [](const SO2 &a, const SO2 &b) -> SO2 { return a * b; }, py::is_operator())
        .def(
            "__imul__",
            [](SO2 &a, const SO2 &b) -> SO2 & {
                a *= b;
                return a;
            },
            py::is_operator());
}
}  // namespace robot::liegroups
