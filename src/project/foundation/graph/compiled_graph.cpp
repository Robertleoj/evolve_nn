#include "foundation/graph/compiled_graph.hpp"
#include "foundation/utils.hpp"

namespace foundation {

namespace graph {


  CompiledGraph::CompiledGraph(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
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
        stored_parameters[idx] = register_parameter(
          ss.str(), 
          torch::randn(1) / 1.24
        );
      }

      if (std::dynamic_pointer_cast<OperatorNode>(node)) {
        std::stringstream ss;
        ss << "mod_" << idx;
        stored_modules[idx] = register_module(ss.str(), std::dynamic_pointer_cast<OperatorNode>(node)->get_op());
      }

      idx++;
    }
  }

  std::vector<torch::Tensor> CompiledGraph::forward(std::vector<torch::Tensor> inputs) {

    std::vector<torch::Tensor> node_outputs;

    for (int i = 0; i < nodes.size(); i++) {

      if (std::dynamic_pointer_cast<InputNode>(nodes[i])) {
        // get index of input_idx in input_nodes
        int input_idx = node_to_input_idx[i];
        node_outputs.push_back(inputs[input_idx]);
        continue;
      }

      if (std::dynamic_pointer_cast<ParameterNode>(nodes[i])) {
        node_outputs.push_back(stored_parameters[i]);
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

        node_outputs.push_back(stored_modules[i]->forward(node_inputs));
      }
    }

    std::vector<torch::Tensor> outputs;
    for (int i : output_nodes) {
      outputs.push_back(node_outputs[i]);
    }

    return outputs;
  }




CompiledGraphWrapper::CompiledGraphWrapper(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
                    std::vector<int> input_order, std::vector<int> output_order)
    : compiled_graph(nodes_topsorted, rev_adj_list, input_order, output_order) {}


std::vector<py::array> CompiledGraphWrapper::forward(std::vector<py::array> inputs) {
    auto tensor_inputs = from_numpy_vec(inputs);

    auto tensor_outputs = compiled_graph.forward(tensor_inputs);
    auto numpy_outputs = to_numpy_vec(tensor_outputs);

    return numpy_outputs;
}

}
}