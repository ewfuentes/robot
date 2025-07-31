#include "experimental/learn_descriptors/translation_prior.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::experimental::learn_descriptors {
PYBIND11_MODULE(translation_prior_python, m) {
    py::class_<TranslationPrior>(m, "TranslationPrior")
        // Default constructor
        .def(py::init<>())

        // Constructor with only translation
        .def(py::init([](const Eigen::Vector3d& t) {
            TranslationPrior prior;
            prior.translation = t;
            prior.covariance = Eigen::Matrix3d::Identity();  // default covariance
            return prior;
        }))

        // Constructor with translation and covariance
        .def(py::init([](const Eigen::Vector3d& t, const Eigen::Matrix3d& cov) {
            TranslationPrior prior;
            prior.translation = t;
            prior.covariance = cov;
            return prior;
        }))

        // Expose fields as read/write properties
        .def_readwrite("translation", &TranslationPrior::translation)
        .def_readwrite("covariance", &TranslationPrior::covariance);
}
}  // namespace robot::experimental::learn_descriptors