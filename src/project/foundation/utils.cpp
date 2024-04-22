#include "foundation/utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace foundation {

torch::Tensor from_numpy(py::array inp) {

  auto dtype = inp.dtype();
  torch::Dtype torch_dtype;
  if (dtype.is(py::dtype::of<float>())) {
    torch_dtype = torch::kFloat32;
  } else if (dtype.is(py::dtype::of<double>())) {
    torch_dtype = torch::kFloat64;
  } else if (dtype.is(py::dtype::of<int>())) {
    torch_dtype = torch::kInt32;
  } else {
    throw std::runtime_error("Unsupported data type");
  }

  std::vector<int64_t> shape(inp.ndim());
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = inp.shape(i);
  }

  auto tensor = torch::from_blob(inp.mutable_data(), shape, torch_dtype);

  return tensor.clone();
}

std::vector<torch::Tensor> from_numpy_vec(std::vector<py::array> inp) {
  std::vector<torch::Tensor> out;
  for (auto &item : inp) {
    out.push_back(from_numpy(item));
  }
  return out;
}

py::array to_numpy(torch::Tensor inp) {
  // Ensure the Tensor is in CPU memory and is contiguous
  inp = inp.contiguous();

  // Detect the data type of the tensor
  py::dtype dtype;
  switch (inp.scalar_type()) {
  case torch::kDouble:
    dtype = py::dtype::of<double>();
    break;
  case torch::kFloat:
    dtype = py::dtype::of<float>();
    break;
  case torch::kInt32:
    dtype = py::dtype::of<int>();
    break;
  default:
    throw std::runtime_error("Unsupported tensor dtype");
  }

  // Calculate the strides for the numpy array in bytes
  std::vector<ssize_t> strides(inp.strides().size());
  for (size_t i = 0; i < strides.size(); ++i) {
    strides[i] = inp.strides()[i] * inp.element_size();
  }

  // Create the numpy array with no additional copying
  return py::array(dtype, inp.sizes(), strides, inp.data_ptr());
}

std::vector<py::array> to_numpy_vec(std::vector<torch::Tensor> &inp) {
  std::vector<py::array> out;
  for (auto &item : inp) {
    out.push_back(to_numpy(item));
  }
  return out;
}
} // namespace foundation
