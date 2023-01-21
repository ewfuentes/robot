
#include "experimental/pokerbots/evaluate_mass_hands.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::experimental::pokerbots {
PYBIND11_MODULE(evaluate_mass_hands_python, m) {
    py::module hand_evaluator = py::module::import("hand_evaluator_python");
    m.def("evaluate_mass_hands", &evaluate_mass_hands);
}

}  // namespace robot::experimental::pokerbots
