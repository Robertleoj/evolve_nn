#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace foundation {

torch::Tensor from_numpy(py::array inp);

std::vector<torch::Tensor> from_numpy_vec(std::vector<py::array> inp);

py::array to_numpy(torch::Tensor inp);

std::vector<py::array> to_numpy_vec(std::vector<torch::Tensor> &inp);

} // namespace foundation
