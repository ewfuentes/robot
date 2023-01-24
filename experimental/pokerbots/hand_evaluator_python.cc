

#include <ios>
#include <sstream>
#include <string>

#include "experimental/pokerbots/hand_evaluator.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::pokerbots {
PYBIND11_MODULE(hand_evaluator_python, m) {
    m.doc() = "Hand Evaluator";

    py::class_<ExpectedStrengthResult>(m, "ExpectedStrengthResult")
        .def_readonly("strength", &ExpectedStrengthResult::strength)
        .def_readonly("num_evaluations", &ExpectedStrengthResult::num_evaluations)
        .def("__repr__", [](const ExpectedStrengthResult &obj) {
            return "<ExpectedStrengthResult strength=" + std::to_string(obj.strength) +
                   " num_evaluations=" + std::to_string(obj.num_evaluations) + ">";
        });

    py::class_<StrengthPotentialResult>(m, "StrengthPotentialResult")
        .def_readonly("strength", &StrengthPotentialResult::strength)
        .def_readonly("strength_potential", &StrengthPotentialResult::strength_potential)
        .def_readonly("positive_potential", &StrengthPotentialResult::positive_potential)
        .def_readonly("negative_potential", &StrengthPotentialResult::negative_potential)
        .def_readonly("num_evaluations", &StrengthPotentialResult::num_evaluations)
        .def("__repr__", [](const StrengthPotentialResult &obj) {
            return "<StrengthPotentialResult strength=" + std::to_string(obj.strength) +
                   " strength_potential=" + std::to_string(obj.strength_potential) +
                   " negative_potential=" + std::to_string(obj.negative_potential) +
                   " positive_potential=" + std::to_string(obj.positive_potential) +
                   " num_evaluations=" + std::to_string(obj.num_evaluations) + ">";
        });

    py::class_<HandDistributionResult>(m, "HandDistributionResult")
        .def_readonly("distribution", &HandDistributionResult::distribution)
        .def_readonly("num_board_rollouts", &HandDistributionResult::num_board_rollouts)
        .def("__repr__", [](const HandDistributionResult &obj) {
            std::ostringstream s;
            s << "<HandDistributionResult distribution=[";
            for (const auto &count : obj.distribution) {
                s << count << ",";
            }
            s.seekp(-1, std::ios_base::end);
            s << "] num_board_rollouts=" << obj.num_board_rollouts << ">";
            return s.str();
        });

    m.def("estimate_hand_distribution",
          py::overload_cast<const std::string &, const std::string &, const int,
                            const std::optional<int>, const std::optional<int>,
                            const std::optional<double>>(&estimate_hand_distribution),
          "hand_str"_a, "board_str"_a, "num_bins"_a, "num_board_rollouts"_a = std::nullopt,
          "max_additional_cards"_a = std::nullopt, "timeout_s"_a = std::nullopt);
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
