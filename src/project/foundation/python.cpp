#include "../../../include/foundation/example.hpp"
#include <torch/torch.h>
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

  m.def("tensor_test", []() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    std::cout << "Tensor size: " << tensor.sizes() << std::endl;
  });

}
