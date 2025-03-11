
#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace robot {
template <typename T>
struct PlotSignal {
    T x;
    T y;
    std::string label = "";
    std::string marker = "-";
};

template <typename T>
void plot(const std::vector<PlotSignal<T>> &signals, const bool block = true);

void contourf(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const Eigen::MatrixXd &data, const bool block=true);

extern template
void plot(const std::vector<PlotSignal<std::vector<double>>> &signals, const bool block);

extern template
void plot(const std::vector<PlotSignal<Eigen::VectorXd>> &signals, const bool block);

}  // namespace robot
