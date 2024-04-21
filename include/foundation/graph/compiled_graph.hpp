#pragma once
#include "foundation/graph/nodes.hpp"
#include "foundation/utils.hpp"
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/torch.h>

namespace py = pybind11;

namespace foundation {
namespace graph {

class CompiledGraph : public torch::nn::Module {

public:
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<std::vector<int>> rev_adj_list;
  // mapping from node index to input index
  std::map<int, int> node_to_input_idx;
  std::vector<int> output_nodes;
  std::map<int, torch::Tensor> parameters;
  std::map<int, std::shared_ptr<OpNodeMod>> modules;

  CompiledGraph(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
                std::vector<int> input_order, std::vector<int> output_order)
      : nodes(nodes_topsorted), rev_adj_list(rev_adj_list), output_nodes(output_order) {

    for (auto i = 0; i < input_order.size(); i++) {
      node_to_input_idx[input_order[i]] = i;
    }

    int idx = 0;
    for (auto &node : nodes) {

      if (std::dynamic_pointer_cast<ParameterNode>(node)) {
        std::stringstream ss;
        ss << "param_" << idx;
        parameters[idx] = register_parameter(ss.str(), torch::randn(1));
      }

      if (std::dynamic_pointer_cast<OperatorNode>(node)) {
        std::stringstream ss;
        ss << "mod_" << idx;
        modules[idx] = register_module(ss.str(), std::dynamic_pointer_cast<OperatorNode>(node)->get_op());
      }

      idx++;
    }
  }

  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs) {

    std::vector<torch::Tensor> node_outputs;

    for (int i = 0; i < nodes.size(); i++) {
      std::cout << "forwarding node " << i << std::endl;

      if (std::dynamic_pointer_cast<InputNode>(nodes[i])) {
        // get index of input_idx in input_nodes
        int input_idx = node_to_input_idx[i];
        node_outputs.push_back(inputs[input_idx]);
        continue;
      }

      if (std::dynamic_pointer_cast<ParameterNode>(nodes[i])) {
        node_outputs.push_back(parameters[i]);
        continue;
      }

      if (std::dynamic_pointer_cast<OutputNode>(nodes[i])) {
        int inp_idx = rev_adj_list[i][0];
        node_outputs.push_back(node_outputs[inp_idx]);
        continue;
      }

      if (std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
        std::vector<torch::Tensor> node_inputs;

        for (int j : rev_adj_list[i]) {
          node_inputs.push_back(node_outputs[j]);
        }

        if (std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
          node_outputs.push_back(modules[i]->forward(node_inputs));
        }
      }
    }

    std::vector<torch::Tensor> outputs;
    for (int i : output_nodes) {
      outputs.push_back(node_outputs[i]);
    }

    return outputs;
  }
};

class CompiledGraphWrapper {
public:
  CompiledGraphWrapper(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
                       std::vector<int> input_order, std::vector<int> output_order)
      : compiled_graph(nodes_topsorted, rev_adj_list, input_order, output_order) {}

  CompiledGraph compiled_graph;

  std::vector<py::array> forward(std::vector<py::array> inputs) {
    // std::cout << "getting numpy inputs" << std::endl;
    auto tensor_inputs = from_numpy(inputs);

    // std::cout << "forwarding" << std::endl;
    auto tensor_outputs = compiled_graph.forward(tensor_inputs);

    // std::cout << "getting numpy outputs" << std::endl;
    auto numpy_outputs = to_numpy(tensor_outputs);

    // std::cout << "returning numpy outputs" << std::endl;
    return numpy_outputs;
  }
};

} // namespace graph
} // namespace foundation
