#include "foundation/graph/nodes.hpp"
#include <iostream>

namespace foundation {
namespace graph {

torch::Tensor AddMod::forward(std::vector<torch::Tensor> inputs) {
  // Stack all tensors along a new dimension (e.g., 0)
  auto stacked_tensors = torch::stack(inputs, 0);

  // Sum across the new dimension to combine all tensors
  return torch::sum(stacked_tensors, 0);
}

std::shared_ptr<OpNodeMod> AddNode::get_op() { return std::make_shared<AddMod>(); }

// NegMod implementation
torch::Tensor NegMod::forward(std::vector<torch::Tensor> inputs) { return -inputs[0]; }

std::shared_ptr<OpNodeMod> NegNode::get_op() { return std::make_shared<NegMod>(); }

// ProdMod implementation
torch::Tensor ProdMod::forward(std::vector<torch::Tensor> inputs) {
  torch::Tensor product = inputs[0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    product = product * inputs[i];
  }
  return product;
}

std::shared_ptr<OpNodeMod> ProdNode::get_op() { return std::make_shared<ProdMod>(); }

// GELUMod implementation
torch::Tensor GELUMod::forward(std::vector<torch::Tensor> inputs) { return torch::gelu(inputs[0]); }

std::shared_ptr<OpNodeMod> GELUNode::get_op() { return std::make_shared<GELUMod>(); }

// LogMod implementation
torch::Tensor LogMod::forward(std::vector<torch::Tensor> inputs) { return torch::log(inputs[0]); }

std::shared_ptr<OpNodeMod> LogNode::get_op() { return std::make_shared<LogMod>(); }

// ExpMod implementation
torch::Tensor ExpMod::forward(std::vector<torch::Tensor> inputs) { return torch::exp(inputs[0]); }

std::shared_ptr<OpNodeMod> ExpNode::get_op() { return std::make_shared<ExpMod>(); }

} // namespace graph
} // namespace foundation
