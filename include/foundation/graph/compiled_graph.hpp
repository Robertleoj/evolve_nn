#pragma once
#include <map>
#include <string>
#include <torch/torch.h>
#include "foundation/graph/nodes.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace foundation {
namespace graph{


class CompiledGraph : public torch::nn::Module {

public:
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::vector<int>> rev_adj_list;
    // mapping from node index to input index
    std::map<int, int> node_to_input_idx;
    std::vector<int> output_nodes;
    std::map<int, torch::Tensor> parameters;
    std::map<int, std::shared_ptr<OpNodeMod>> modules;

    CompiledGraph(
        std::vector<std::shared_ptr<Node>> nodes_topsorted,
        std::vector<std::vector<int>> rev_adj_list,
        std::vector<int> input_order,
        std::vector<int> output_order
    ): nodes(nodes_topsorted), rev_adj_list(rev_adj_list), output_nodes(output_order){

        for(auto i = 0; i < input_order.size(); i++) {
            node_to_input_idx[input_order[i]] = i;
        }

        int idx = 0;
        for (auto &node: nodes) {

            if (std::dynamic_pointer_cast<ParameterNode>(node)) {
                std::stringstream ss;
                ss << "param_" << idx;
                parameters[idx] = register_parameter(
                    ss.str(), 
                    torch::randn(1)
                );
            }

            if(std::dynamic_pointer_cast<OperatorNode>(node)) {
                std::stringstream ss;
                ss << "mod_" << idx;
                modules[idx] = register_module(
                    ss.str(),
                    std::dynamic_pointer_cast<OperatorNode>(node)->get_op()
                );
            }

            idx++;
        }
    }

    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs) {

        std::vector<torch::Tensor> node_outputs;

        for (int i = 0; i < nodes.size(); i++){

            if(std::dynamic_pointer_cast<InputNode>(nodes[i])) {
                // get index of input_idx in input_nodes
                int input_idx = node_to_input_idx[i];
                node_outputs.push_back(inputs[input_idx]);
                continue;
            }

            if(std::dynamic_pointer_cast<ParameterNode>(nodes[i])) {
                node_outputs.push_back(parameters[i]);
                continue;
            }

            if(std::dynamic_pointer_cast<OutputNode>(nodes[i])) {
                int inp_idx = rev_adj_list[i][0];
                node_outputs.push_back(node_outputs[inp_idx]);
                continue;
            }

            if(std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
                std::vector<torch::Tensor> inputs;
                for (int j: rev_adj_list[i]) {
                    inputs.push_back(node_outputs[j]);
                }

                if (std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
                    node_outputs[i] = modules[i]->forward(inputs);
                }
            }
        }

        std::vector<torch::Tensor> outputs;
        for (int i: output_nodes) {
            outputs.push_back(node_outputs[i]);
        }

        return outputs;
    } 
};

class CompiledGraphWrapper {
public:
    CompiledGraphWrapper(
        std::vector<std::shared_ptr<Node>> nodes_topsorted,
        std::vector<std::vector<int>> rev_adj_list,
        std::vector<int> input_order,
        std::vector<int> output_order
    ): compiled_graph(nodes_topsorted, rev_adj_list, input_order, output_order) {}

    CompiledGraph compiled_graph;

    std::vector<py::array> forward(std::vector<py::array> inputs) {
        std::vector<torch::Tensor> inputs_torch;
        inputs_torch.reserve(inputs.size());

        // Convert inputs to torch::Tensor
        for (auto &inp: inputs) {
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

                    // Adjust shape conversion
            std::vector<int64_t> shape(inp.ndim());
            for (size_t i = 0; i < shape.size(); ++i) {
                shape[i] = inp.shape(i);
            }

            inputs_torch.push_back(torch::from_blob(
                inp.mutable_data(), 
                shape,
                torch_dtype)
            );
        }

        // Assume compiled_graph is an instance of some model that can process this
        std::vector<torch::Tensor> outputs = compiled_graph.forward(inputs_torch);

        std::vector<py::array> outputs_np;
        outputs_np.reserve(outputs.size());

        // Convert outputs to py::array
        for (auto &out: outputs) {
            py::dtype dtype = py::dtype::of<float>(); // Defaulting to float; change as necessary
            if (out.scalar_type() == torch::kFloat64) {
                dtype = py::dtype::of<double>();
            } else if (out.scalar_type() == torch::kInt32) {
                dtype = py::dtype::of<int>();
            }
            outputs_np.push_back(py::array(dtype, out.sizes(), out.strides(), out.data_ptr()));
        }

        return outputs_np;
    }

};

}
}