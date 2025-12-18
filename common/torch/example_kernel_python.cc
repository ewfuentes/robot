#include <torch/extension.h>  // PyTorch extension API with pybind11 support
#include "common/torch/example_kernel.hh"

namespace robot::torch {

PYBIND11_MODULE(example_kernel_python, m) {
    m.doc() = "Pytorch Extension Example";
    m.def("square", &square, "a function to square a torch tensor");
}

}  // namespace robot::torch
