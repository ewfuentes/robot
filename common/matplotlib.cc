
#include "common/matplotlib.hh"

#include <cstdlib>
#include <limits>

#include "pybind11/embed.h"
#include "pybind11/stl.h"

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

void plot(const std::vector<double> &x, const std::vector<double> &y) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.home = to_wchar(CPP_PYTHON_HOME);
    config.pathconfig_warnings = 1;
    config.program_name = to_wchar(CPP_PYVENV_LAUNCHER);
    config.pythonpath_env = to_wchar(CPP_PYTHON_PATH);
    config.user_site_directory = 0;
    py::scoped_interpreter guard{&config};

    py::module_ plt = py::module_::import("matplotlib.pyplot");

    plt.attr("figure")();
    plt.attr("plot")(x, y);
    plt.attr("show")("block"_a = false);
}
}  // namespace robot
