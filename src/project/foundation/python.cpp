#include "../../../include/foundation/example.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iostream>
#include <string>


namespace py = pybind11;

using namespace foundation;

PYBIND11_MODULE(foundation, m) {
  m.doc() = R"pbdoc(
        Bindings to the foundation.
        ---------------------------
    )pbdoc";

  m.def("add", add, "Add two numbers", py::arg("a"), py::arg("b"), R"pbdoc(
        Add two numbers
  )pbdoc");
}
