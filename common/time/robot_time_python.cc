
#include "common/time/robot_time.hh"
#include "pybind11/chrono.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

#include <iomanip>
#include <sstream>

namespace py = pybind11;

namespace robot::time {
PYBIND11_MODULE(robot_time_python, m) {
    m.doc() = "robot time";

    m.def("current_robot_time", &current_robot_time);
    m.def("as_duration", &as_duration);

    py::class_<RobotTimestamp>(m, "RobotTimestamp")
        .def(py::init<>())
        .def("time_since_epoch", &RobotTimestamp::time_since_epoch)
        .def_property_readonly_static("MIN", &RobotTimestamp::min)
        .def_property_readonly_static("MAX", &RobotTimestamp::max)
        .def(py::self - py::self)
        .def(py::self - RobotTimestamp::duration())
        .def(py::self + RobotTimestamp::duration())
        .def(RobotTimestamp::duration() + py::self)
        .def(RobotTimestamp::duration() - py::self)
        .def(py::self -= RobotTimestamp::duration())
        .def(py::self += RobotTimestamp::duration())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def("__repr__", [](const RobotTimestamp &time)
             { 
             std::ostringstream ss;
             ss << std::fixed << std::setprecision(9) << std::chrono::duration<double>(time.time_since_epoch()).count();
             return ss.str();
             });
}
}  // namespace robot::time
