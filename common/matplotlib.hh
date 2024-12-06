
#pragma once

#include <string>
#include <vector>

namespace robot {
struct PlotSignal {
    std::vector<double> x;
    std::vector<double> y;
    std::string label = "";
    std::string marker = "-";
};

void plot(const std::vector<PlotSignal> &signals, const bool block = true);

}  // namespace robot
