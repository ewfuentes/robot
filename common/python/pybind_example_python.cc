
#include "common/python/test_function.hh"
#include "pybind11/pybind11.h"

namespace robot::python {

PYBIND11_MODULE(pybind_example_python, m) {
    m.doc() = "PyBind11 Example";
    m.def("add", &add, "a function to add two numbers");
}

}  // namespace robot::python
