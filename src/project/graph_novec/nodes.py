import torch
import torch.nn as nn
import math

class Node(nn.Module):
    """Base class for all nodes in the graph.

    Attributes:
        name: Name of the node. Must be unique for all node classes.
    """

    def __init__(self) -> None:
        """Create a node."""
        super().__init__()

    name: str


# Data nodes
class DataNode(Node):
    """A node that represents data."""

class InputNode(DataNode):
    """A node that represents input data."""
    name = "input"


class ParameterNode(DataNode):
    """A node that represents a learnable parameter."""
    name = "parameter"


class OutputNode(DataNode):
    """A node that represents output data."""
    name = "output"


# operator nodes
class OperatorNode(Node):
    """A node that represents an operation.

    Attributes:
        input_shapes: Shapes of the input tensors.
        output_shapes: Shapes of the output tensors.
    """

    n_inputs: tuple[int, int | None]

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the operation.

        Args:
            inputs: List of input tensors.
        """
        raise NotImplementedError

class AddNode(OperatorNode):
    """A node that adds two or more tensors."""

    name = "add"
    n_inputs = (1, None)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the addition operation."""
        return sum(inputs)


class ProdNode(OperatorNode):
    """A node that multiplies two or more tensors."""

    name = "prod"
    n_inputs = (1, None)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the multiplication operation."""
        return math.prod(inputs)


class GELUNode(OperatorNode):
    """A node that applies the ReLu function."""

    name = "gelu"
    n_inputs = (1, 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the ReLu operation."""
        return nn.functional.gelu(inputs[0])

class LogNode(OperatorNode):
    """A node that applies the log function."""

    name = "log"
    n_inputs = (1, 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the log operation."""
        return torch.log(inputs[0])

class ExpNode(OperatorNode):
    """A node that applies the exp function."""

    name = "exp"
    n_inputs = (1, 1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the exp operation."""
        return torch.exp(inputs[0])


operator_nodes = [
    AddNode,
    ProdNode,
    GELUNode,
    LogNode,
    ExpNode,
]

node_map = {
    node_type.name: node_type
    for node_type in [InputNode, OutputNode, ParameterNode, AddNode, ProdNode, GELUNode, LogNode, ExpNode]
}


def node_from_name(name: str) -> Node:
    """Create a node from a dictionary representation.

    Args:
        spec: Dictionary representation of the node.

    Returns:
        Node: The node.
    """
    return node_map[name]()
