
#include "common/matplotlib.hh"

#include <cstdlib>
#include <limits>

#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot {
namespace {
wchar_t *to_wchar(const char *str) {
    const size_t num_chars = std::mbstowcs(nullptr, str, 0) + 1;
    if (num_chars == std::numeric_limits<size_t>::max()) {
        return nullptr;
    }
    wchar_t *out = static_cast<wchar_t *>(malloc(num_chars * sizeof(wchar_t)));

    std::mbstowcs(out, str, num_chars);

    return out;
}
}  // namespace

template <typename T>
void plot(const std::vector<PlotSignal<T>> &signals, const bool block) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.home = to_wchar(CPP_PYTHON_HOME);
    config.pathconfig_warnings = 1;
    config.program_name = to_wchar(CPP_PYVENV_LAUNCHER);
    config.pythonpath_env = to_wchar(CPP_PYTHON_PATH);
    config.user_site_directory = 0;
    py::scoped_interpreter guard{&config};

    py::module_ mpl = py::module_::import("matplotlib");
    mpl.attr("use")("GTK3Agg");
    py::module_ plt = py::module_::import("matplotlib.pyplot");

    plt.attr("figure")();
    for (const auto &signal : signals) {
        plt.attr("plot")(signal.x, signal.y, signal.marker, "label"_a = signal.label);
    }
    plt.attr("legend")();
    plt.attr("show")("block"_a = block);
}

void contourf(const Eigen::VectorXd &x, const Eigen::VectorXd &y,
              const Eigen::MatrixXd &data, const bool block) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.home = to_wchar(CPP_PYTHON_HOME);
    config.pathconfig_warnings = 1;
    config.program_name = to_wchar(CPP_PYVENV_LAUNCHER);
    config.pythonpath_env = to_wchar(CPP_PYTHON_PATH);
    config.user_site_directory = 0;
    py::scoped_interpreter guard{&config};

    py::module_ mpl = py::module_::import("matplotlib");
    mpl.attr("use")("GTK3Agg");
    py::module_ plt = py::module_::import("matplotlib.pyplot");

    plt.attr("figure")();
    plt.attr("contourf")(x, y, data, Eigen::VectorXd::LinSpaced(8, -3, 4));
    plt.attr("colorbar")();
    plt.attr("show")("block"_a = block);
}

template
void plot(const std::vector<PlotSignal<std::vector<double>>> &signals, const bool block);

template
void plot(const std::vector<PlotSignal<Eigen::VectorXd>> &signals, const bool block);

}  // namespace robot
