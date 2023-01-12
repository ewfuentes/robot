
#include "omp/CardRange.h"
#include "omp/EquityCalculator.h"
#include "pybind11/pybind11.h"
#include "vector"

namespace robot::experimental::pokerbots {
namespace {
struct EquityResult {
    double equity;
    uint64_t num_evaluations;
};

EquityResult evaluate_hand(const std::string &hand, const std::string &opponent_hand_str,
                           const std::string &board_str) {
    const std::vector<omp::CardRange> ranges = {{hand}, {opponent_hand_str}};
    const uint64_t board = omp::CardRange::getCardMask(board_str);
    omp::EquityCalculator calculator;
    calculator.setTimeLimit(0.02);
    calculator.start(ranges, board);
    calculator.wait();
    const auto results = calculator.getResults();

    return {
        .equity = results.equity[0],
        .num_evaluations = results.hands,
    };
}
}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(hand_evaluator_python, m) {
    m.doc() = "Hand Evaluator";

    py::class_<EquityResult>(m, "EquityResult")
        .def_readonly("equity", &EquityResult::equity)
        .def_readonly("num_evaluations", &EquityResult::num_evaluations);

    m.def("evaluate_hand", &evaluate_hand);
}
}  // namespace robot::experimental::pokerbots
