#pragma once

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include "torch/torch.h"

namespace foundation {
namespace graph {

// Base class for all nodes in the graph
class Node {
public:
    virtual ~Node() = default;
};

// Data nodes
class DataNode : public Node {};

class InputNode : public DataNode {};

class ResponseInputNode : public DataNode {};

class OutputNode : public DataNode {};

class LossOutputNode : public DataNode {};

class ParameterNode : public DataNode {};

class OpNodeMod: public torch::nn::Module {
public:
    virtual torch::Tensor forward(std::vector<torch::Tensor> inputs) {
        return torch::Tensor();
    };
};

// Operator nodes
class OperatorNode : public Node {
public:
    virtual std::shared_ptr<OpNodeMod> get_op() {
        return nullptr;
    };
};


class AddMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class AddNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

class NegMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class NegNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

class ProdMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class ProdNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

class GELUMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class GELUNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

class LogMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class LogNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

class ExpMod : public OpNodeMod {
public:
    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;
};

class ExpNode : public OperatorNode {
public:
    std::shared_ptr<OpNodeMod> get_op() override;
};

} // namespace graph
} // namespace foundation

