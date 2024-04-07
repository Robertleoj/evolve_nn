import torch
import torch.nn as nn


class Node(nn.Module):
    """Base class for all nodes in the graph.

    Attributes:
        name: Name of the node. Must be unique for all node classes.
    """

    name: str
    shape: tuple[int, ...]

    def get_spec(self) -> dict:
        """Return a dictionary representation of the node."""
        raise NotImplementedError

    def from_spec(self, spec: dict) -> "Node":
        """Create a node from a dictionary representation.

        Args:
            spec: Dictionary representation of the node.

        Returns:
            Node: The node.
        """
        raise NotImplementedError


# Data nodes
class DataNode(Node):
    """A node that represents data.

    Attributes:
        shape: Shape of the data.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Create a data node.

        Args:
            shape: Shape of the data.
        """
        super().__init__()
        self.shape = shape

    def get_spec(self) -> dict:
        """Return a dictionary representation of the node.

        Returns:
            dict: Dictionary representation of the node.
        """
        return {"shape": self.shape, "name": self.name}

    @classmethod
    def from_spec(cls, spec: dict) -> "DataNode":
        """Create a data node from a dictionary representation.

        Args:
            spec: Dictionary representation of the node.

        Returns:
            DataNode: The data node.
        """
        return cls(spec["shape"])


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

    input_shapes: tuple[tuple[int, ...]]

    def __init__(self, input_shapes: tuple[tuple[int, ...]], shape: tuple[int, ...]) -> None:
        """Create an operator node.

        Args:
            input_shapes: Shapes of the input tensors.
            output_shapes: Shapes of the output tensors.
        """
        super().__init__()
        self.shape = shape
        self.input_shapes = input_shapes

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the operation.

        Args:
            inputs: List of input tensors.
        """
        raise NotImplementedError

    def get_spec(self) -> dict:
        """Return a dictionary representation of the node.

        Returns:
            dict: Dictionary representation of the node.
        """
        return {"input_shapes": self.input_shapes, "shape": self.shape, "name": self.name}

    @classmethod
    def from_spec(cls, spec: dict) -> "OperatorNode":
        """Create an operator node from a dictionary representation."""
        return cls(spec["input_shapes"], spec["shape"])


class AddNode(OperatorNode):
    """A node that adds two or more tensors."""

    name = "add"

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the addition operation."""
        return sum(inputs)


class MatmulNode(OperatorNode):
    """A node that performs matrix multiplication."""

    name = "matmul"

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the matrix multiplication operation."""
        return torch.matmul(*inputs)


class ReLuNode(OperatorNode):
    """A node that applies the ReLu function."""

    name = "relu"

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Perform the ReLu operation."""
        return torch.relu(inputs[0])


node_map = {
    node_type.name: node_type
    for node_type in [InputNode, OutputNode, ParameterNode, AddNode, MatmulNode, ReLuNode]
}


def node_from_spec(spec: dict) -> Node:
    """Create a node from a dictionary representation.

    Args:
        spec: Dictionary representation of the node.

    Returns:
        Node: The node.
    """
    return node_map[spec["name"]].from_spec(spec)
