#include <torch/extension.h>
#include <torch/library.h>

#include <fstream>
#include <iostream>
#include <string>

// namespace py = pybind11;

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) { return a + b; }

TORCH_LIBRARY(foundation, m) {
  // m.def("add", add, "Add two numbers", py::arg("a"), py::arg("b"), R"pbdoc(
  // Add two numbers
  // )pbdoc");
  // m.def("add_tensors", &add_tensors, "A function that adds two tensors");

  m.def("add_tensors(Tensor a, Tensor b) -> Tensor");
  m.impl("add_tensors", c10::DispatchKey::CPU, &add_tensors);
}
