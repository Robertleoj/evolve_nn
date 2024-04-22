from __future__ import annotations

import math
from typing import Any

import project.foundation.graph as cpp_graph
import project.graph.compiled as compiled_
import project.graph.graph as graph_
import torch
import torch.nn as nn


class Node:
    """Base class for all nodes in the graph."""

    name: str


# Data nodes
class DataNode(Node):
    """A node that represents data."""


class InputNode(DataNode):
    """A node that represents input data."""

    name = "input"


class ResponseInputNode(DataNode):
    """A node that represents response input data."""

    name = "response_input"


class OutputNode(DataNode):
    """A node that represents output data."""

    name = "output"


class LossOutputNode(DataNode):
    """A node that represents the loss output data."""

    name = "loss_output"


class ParameterNode(DataNode):
    """A node that represents a learnable parameter."""

    name = "parameter"


# operator nodes
class OperatorNode(Node):
    """A node that represents an operation."""

    n_inputs: tuple[int, int | None]
    inputs_ordered: bool = False

    def get_op(self) -> nn.Module:
        """Perform the operation."""
        raise NotImplementedError


class AddMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return sum(inp)  # type: ignore


class AddNode(OperatorNode):
    """A node that adds two or more tensors."""

    name = "add"
    n_inputs = (2, None)

    def get_op(self) -> nn.Module:
        """Perform the addition operation."""
        return AddMod()


class NegMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return -inp[0]


class NegNode(OperatorNode):
    """A node that negates a tensor."""

    name = "neg"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the negation operation."""
        return NegMod()


class ProdMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return math.prod(inp)


class ProdNode(OperatorNode):
    """A node that multiplies two or more tensors."""

    name = "prod"
    n_inputs = (2, None)

    def get_op(self) -> nn.Module:
        """Perform the multiplication operation."""
        return ProdMod()


class GELUMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return nn.functional.gelu(inp[0])


class GELUNode(OperatorNode):
    """A node that applies the ReLu function."""

    name = "GELU"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the ReLu operation."""
        return GELUMod()


class LogMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return torch.log(inp[0])


class LogNode(OperatorNode):
    """A node that applies the log function."""

    name = "log"

    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the log operation."""
        return LogMod()


class ExpMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return torch.exp(inp[0])


class ExpNode(OperatorNode):
    """A node that applies the exp function."""

    name = "exp"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the exp operation."""
        return ExpMod()


class SquareMod(nn.Module):
    def __call__(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return inp[0] ** 2


class SquareNode(OperatorNode):
    """A node that applies the exp function."""

    name = "square"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the exp operation."""
        return SquareMod()


class SubGraphNode(OperatorNode):
    """A node that represents a subgraph."""

    name = "graph"
    subgraph: graph_.Graph

    def __init__(self, subgraph: graph_.Graph) -> None:
        num_inputs = len(subgraph.ordered_input_nodes)
        self.subgraph = subgraph
        self.n_inputs = (num_inputs, num_inputs)

    def get_op(self) -> graph_.CompiledGraph:
        return compiled_.SubCompiledGraph.from_graph(self.subgraph)


op_nodes: list[type[OperatorNode]] = [
    AddNode,
    NegNode,
    ProdNode,
    SquareNode,
    # GELUNode,
    # ExpNode,
    SubGraphNode,
]

data_nodes: list[type[DataNode]] = [
    InputNode,
    OutputNode,
    ParameterNode,
    ResponseInputNode,
    LossOutputNode,
]

op_node_name_to_node: dict[str, type[OperatorNode]] = {node.name: node for node in op_nodes}

data_node_name_to_node: dict[str, type[DataNode]] = {node.name: node for node in data_nodes}

node_name_to_node: dict[str, type[Node]] = {
    **op_node_name_to_node,
    **data_node_name_to_node,
}


def node_from_spec(spec: dict[str, Any]) -> Node:
    """Create a node from a dictionary representation.

    Args:
        spec: Dictionary representation of the node.

    Returns:
        Node: The node.
    """
    node_name = spec["name"]
    node = node_name_to_node[node_name]
    if "args" in spec:
        return node(**spec["args"])
    return node()


def to_cpp_node(node: Node) -> cpp_graph.Node:
    match node.name:
        case "input":
            return cpp_graph.InputNode()
        case "response_input":
            return cpp_graph.ResponseInputNode()
        case "output":
            return cpp_graph.OutputNode()
        case "loss_output":
            return cpp_graph.LossOutputNode()
        case "parameter":
            return cpp_graph.ParameterNode()
        case "add":
            return cpp_graph.AddNode()
        case "neg":
            return cpp_graph.NegNode()
        case "prod":
            return cpp_graph.ProdNode()
        case "GELU":
            return cpp_graph.GELUNode()
        case "log":
            return cpp_graph.LogNode()
        case "exp":
            return cpp_graph.ExpNode()
        case "square":
            return cpp_graph.SquareNode()
        case _:
            raise ValueError(f"Unknown node name: {node.name}")
