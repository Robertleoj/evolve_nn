#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

        auto tensor = torch::from_blob(
            inp.mutable_data(), 
            shape,
            torch_dtype
        );

        return tensor.clone();
    }

    std::vector<torch::Tensor> from_numpy(std::vector<py::array> inp) {
        std::vector<torch::Tensor> out;
        for (auto &item : inp) {
            out.push_back(from_numpy(item.cast<py::array>()));
        }
        return out;
    }


    py::array to_numpy(torch::Tensor inp) {

        auto sizes = inp.sizes();
        std::vector<ssize_t> shape(sizes.begin(), sizes.end());

        auto dtype = py::dtype::of<float>();
        if (inp.scalar_type() == torch::kFloat64) {
            dtype = py::dtype::of<double>();
        } else if (inp.scalar_type() == torch::kInt32) {
            dtype = py::dtype::of<int>();
        }

        auto arr = py::array(
            dtype,
            inp.sizes(),
            inp.strides(),
            inp.data_ptr()
        );

        return arr;
    }

    std::vector<py::array> to_numpy(std::vector<torch::Tensor> inp) {
        std::vector<py::array> out;
        for (auto &item : inp) {
            out.push_back(to_numpy(item));
        }
        return out;
    }
}
