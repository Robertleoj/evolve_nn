#include "foundation/graph/compiled_graph.hpp"
#include "foundation/utils.hpp"

namespace foundation {

namespace graph {

CompiledGraph::CompiledGraph(std::vector<std::shared_ptr<Node>> nodes_topsorted,
                             std::vector<std::vector<int>> rev_adj_list, std::vector<int> input_order,
                             std::vector<int> output_order,
                             std::optional<std::vector<int>> response_order,
                             std::optional<int> loss_node
                      )
    : nodes(nodes_topsorted), rev_adj_list(rev_adj_list), output_nodes(output_order), loss_node(loss_node) {

  for (auto i = 0; i < input_order.size(); i++) {
    node_to_input_idx[input_order[i]] = i;
  }

  if(response_order.has_value()) {
    for (auto i = 0; i < response_order.value().size(); i++) {
      node_to_response_idx[response_order.value()[i]] = i;
    }
  }

  int idx = 0;
  for (auto &node : nodes) {

    if (std::dynamic_pointer_cast<ParameterNode>(node)) {
      std::stringstream ss;
      ss << "param_" << idx;
      stored_parameters[idx] = register_parameter(ss.str(), torch::randn(1) / 1.24);
    }

    if (std::dynamic_pointer_cast<OperatorNode>(node)) {
      std::stringstream ss;
      ss << "mod_" << idx;
      stored_modules[idx] = register_module(ss.str(), std::dynamic_pointer_cast<OperatorNode>(node)->get_op());
    }

    idx++;
  }
}

void CompiledGraph::reset_computation() {
  curr_computation.clear();
  curr_computation.resize(nodes.size());
  for (int i = 0; i < nodes.size(); i++) {
    curr_computation[i] = std::nullopt;
  }
}

std::vector<torch::Tensor> CompiledGraph::forward(std::vector<torch::Tensor> inputs) {

  reset_computation();

  for (int i = 0; i < nodes.size(); i++) {

    if (std::dynamic_pointer_cast<InputNode>(nodes[i])) {
      // get index of input_idx in input_nodes
      int input_idx = node_to_input_idx[i];

      curr_computation[i] = inputs[input_idx];
      continue;
    }

    if (std::dynamic_pointer_cast<ParameterNode>(nodes[i])) {
      curr_computation[i] = stored_parameters[i];
      continue;
    }

    if (std::dynamic_pointer_cast<OutputNode>(nodes[i])) {
      int inp_idx = rev_adj_list[i][0];
      curr_computation[i] = curr_computation[inp_idx];
      continue;
    }

    if (std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
      std::vector<torch::Tensor> node_inputs;

      bool all_exist = true;
      for (int j : rev_adj_list[i]) {
        auto inp = curr_computation[j];

        if(!inp.has_value()){
          all_exist = false;
          break;
        }
        node_inputs.push_back(inp.value());
      }

      if(all_exist) {
        curr_computation[i] = stored_modules[i]->forward(node_inputs);
      }
    }
  }

  std::vector<torch::Tensor> outputs;
  for (int i : output_nodes) {
    auto val = curr_computation[i];
    if(!val.has_value()) {
      throw std::runtime_error("Output node has no value");
    }
    outputs.push_back(val.value());
  }

  return outputs;
}

torch::Tensor CompiledGraph::forward_response(std::vector<torch::Tensor> inputs) {

  if(!node_to_response_idx.size()) {
    throw std::runtime_error("No response nodes in the graph");
  }

  if(!loss_node.has_value()) {
    throw std::runtime_error("No loss node in the graph");
  }

  for(int i = 0; i < nodes.size(); i++) {

    if(std::dynamic_pointer_cast<ResponseInputNode>(nodes[i])) {
      int input_idx = node_to_response_idx[i];
      curr_computation[i] = inputs[input_idx];
    }

    if(curr_computation[i].has_value()) {
      // computed in forward pass
      continue;
    }

    if(std::dynamic_pointer_cast<ParameterNode>(nodes[i])) {
      curr_computation[i] = stored_parameters[i];
      continue;
    }

    if(std::dynamic_pointer_cast<OperatorNode>(nodes[i])) {
      std::vector<torch::Tensor> node_inputs;
      for(int j : rev_adj_list[i]) {
        auto inp = curr_computation[j];
        if(!inp.has_value()) {
          throw std::runtime_error("Operator node has no value");
        }
        node_inputs.push_back(inp.value());
      }
      curr_computation[i] = stored_modules[i]->forward(node_inputs);
      continue;
    }

    if(std::dynamic_pointer_cast<LossOutputNode>(nodes[i])) {
      int inp_idx = rev_adj_list[i][0];
      curr_computation[i] = curr_computation[inp_idx];
    }
  }

  return curr_computation[loss_node.value()].value();
}



// CompiledGraphWrapper
CompiledGraphWrapper::CompiledGraphWrapper(std::vector<std::shared_ptr<Node>> nodes_topsorted,
                                           std::vector<std::vector<int>> rev_adj_list, std::vector<int> input_order,
                                           std::vector<int> output_order,
                                           std::optional<std::vector<int>> response_order,
                                           std::optional<int> loss_node
                                           )
    : compiled_graph(nodes_topsorted, rev_adj_list, input_order, output_order, response_order, loss_node) {}

std::vector<py::array> CompiledGraphWrapper::forward(std::vector<py::array> inputs) {
  auto tensor_inputs = from_numpy_vec(inputs);

  auto tensor_outputs = compiled_graph.forward(tensor_inputs);
  auto numpy_outputs = to_numpy_vec(tensor_outputs);

  return numpy_outputs;
}

py::array CompiledGraphWrapper::forward_response(std::vector<py::array> inputs) {
  auto tensor_inputs = from_numpy_vec(inputs);
  auto tensor_outputs = compiled_graph.forward_response(tensor_inputs);
  return to_numpy(tensor_outputs);
}

} // namespace graph
} // namespace foundation
