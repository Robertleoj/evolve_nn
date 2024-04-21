#pragma once

#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <torch/torch.h>

#include "foundation/graph/nodes.hpp"

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
  std::map<int, torch::Tensor> stored_parameters;
  std::map<int, std::shared_ptr<OpNodeMod>> stored_modules;

  CompiledGraph(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
                std::vector<int> input_order, std::vector<int> output_order);


  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);
};

class CompiledGraphWrapper {
public:
  CompiledGraphWrapper(std::vector<std::shared_ptr<Node>> nodes_topsorted, std::vector<std::vector<int>> rev_adj_list,
                       std::vector<int> input_order, std::vector<int> output_order);

  CompiledGraph compiled_graph;

  std::vector<py::array> forward(std::vector<py::array> inputs);
};

} // namespace graph
} // namespace foundation
