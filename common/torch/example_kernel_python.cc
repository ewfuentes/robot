#include "common/torch/example_kernel.hh"
#include "torch/extension.h"

namespace robot::torch {

PYBIND11_MODULE(example_kernel_python, m) {
    m.doc() = "Pytorch Extension Example";
    m.def("square", &square, "a function to square a torch tensor");
}

}  // namespace robot::torch
