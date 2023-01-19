

#include "experimental/pokerbots/hand_evaluator.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace robot::experimental::pokerbots {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(hand_evaluator_python, m) {
    m.doc() = "Hand Evaluator";

    py::class_<ExpectedStrengthResult>(m, "ExpectedStrengthResult")
        .def_readonly("strength", &ExpectedStrengthResult::strength)
        .def_readonly("num_evaluations", &ExpectedStrengthResult::num_evaluations);

    py::class_<StrengthPotentialResult>(m, "StrengthPotentialResult")
        .def_readonly("strength", &StrengthPotentialResult::strength)
        .def_readonly("strength_potential", &StrengthPotentialResult::strength_potential)
        .def_readonly("positive_potential", &StrengthPotentialResult::positive_potential)
        .def_readonly("negative_potential", &StrengthPotentialResult::negative_potential)
        .def_readonly("num_evaluations", &StrengthPotentialResult::num_evaluations);

    m.def("evaluate_expected_strength", &evaluate_expected_strength, "hand"_a,
          "opponent_hand_str"_a = "random", "board_str"_a, "timeout_s"_a = std::nullopt,
          "num_hands"_a = std::nullopt);
    m.def("evaluate_strength_potential",
          py::overload_cast<const std::string &, const std::string &, const std::optional<int>,
                            const std::optional<double>, const std::optional<int>>(
              &evaluate_strength_potential),
          "hand_str"_a, "board_str"_a, "max_additional_cards"_a = std::nullopt,
          "timeout_s"_a = std::nullopt, "num_hands"_a = std::nullopt);
}
}  // namespace robot::experimental::pokerbots
