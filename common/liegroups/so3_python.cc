#include "common/liegroups/so3.hh"
#include "pybind11/eigen.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::liegroups {

PYBIND11_MODULE(so3_python, m) {
    py::class_<SO3>(m, "SO3")
        .def(py::init<>())
        // Construct from quaternion. Note that we use the (x, y, z, w) convention
        .def_static("from_quat",
                    [](const Eigen::Vector<double, 4> &quat) -> SO3 {
                        const Eigen::Quaterniond eigen_quat(quat(3), quat(0), quat(1), quat(2));
                        return SO3(eigen_quat);
                    })
        .def_static("from_matrix", [](const Eigen::Matrix3d &rot) -> SO3 { return SO3(rot); })
        .def_static("exp",
                    [](const Eigen::Vector3d &tangent_vec) -> SO3 { return SO3::exp(tangent_vec); })
        .def_static("rotX", [](const double theta) -> SO3 { return SO3::rotX(theta); })
        .def_static("rotY", [](const double theta) -> SO3 { return SO3::rotY(theta); })
        .def_static("rotZ", [](const double theta) -> SO3 { return SO3::rotZ(theta); })
        .def("log", &SO3::log)
        .def("inverse", [](const SO3 &a) -> SO3 { return a.inverse(); })
        .def("matrix", &SO3::matrix)
        .def("quaternion",
             [](const SO3 &self) {
                 const Eigen::Quaterniond quat = self.unit_quaternion();
                 return Eigen::Vector<double, 4>{quat.x(), quat.y(), quat.z(), quat.w()};
             })
        .def(py::self * Eigen::Vector3d())
        .def(
            "__matmul__",
            [](const SO3 &a, const Eigen::Matrix3Xd &b) -> Eigen::Matrix3Xd {
                Eigen::Matrix3Xd out(3, b.cols());
                for (int i = 0; i < b.cols(); i++) {
                    out.col(i) = a * b.col(i);
                }
                return out;
            },
            py::is_operator())
        .def(
            "__mul__", [](const SO3 &a, const SO3 &b) -> SO3 { return a * b; }, py::is_operator())
        .def(
            "__imul__",
            [](SO3 &a, const SO3 &b) -> SO3 & {
                a *= b;
                return a;
            },
            py::is_operator());
}
}  // namespace robot::liegroups
