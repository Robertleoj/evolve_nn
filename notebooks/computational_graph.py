# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: project-xOhHZUaJ-py3.10
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn


# %%
class Node(nn.Module):
    name: str

    def get_spec(self) -> dict:
        raise NotImplementedError

    def from_spec(self, spec: dict) -> None:
        raise NotImplementedError

class DataNode(Node):
    name: str
    shape: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def get_spec(self) -> dict:
        return {
            "shape": self.shape,
            "name": self.name
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "DataNode":
        return cls(spec["shape"])

class InputNode(DataNode):
    name = "input"

class OutputNode(DataNode):
    name = "output"

class ParameterNode(DataNode):
    name = "parameter"


class OperatorNode(Node):
    name: str

    input_shapes: tuple[tuple[int, ...]]
    output_shapes: tuple[tuple[int, ...]]

    def __init__(self, input_shapes: tuple[tuple[int, ...]], output_shapes: tuple[tuple[int, ...]]) -> None:
        super().__init__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError

    def get_spec(self) -> dict:
        return {
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "name": self.name
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "OperatorNode":
        return cls(spec["input_shapes"], spec["output_shapes"])



# %%
# Specific operator nodes

class AddNode(OperatorNode):

    name = "add"

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return sum(inputs)

class MatmulNode(OperatorNode):

    name = "matmul"

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.matmul(*inputs)]


class ReLuNode(OperatorNode):

    name = "relu"

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.relu(inputs[0])]


# %%
node_map = {
    "Input": InputNode,
    "Output": OutputNode,
    "Parameter": ParameterNode,
    "Add": AddNode,
    "Matmul": MatmulNode,
    "ReLu": ReLuNode
}


# %%
def get_node(spec: dict): 
    return node_map[spec["name"]].from_spec(spec)


# %%
add_node = get_node({
    "name": "Add",
    "input_shapes": [(1, 2), (1, 2)],
    "output_shapes": [(1, 2)]
})

# %%
add_node([torch.tensor([1, 2]), torch.tensor([3, 4])])


# %%
class Graph:
    nodes: list[Node]
    adjacency_list: list[list[int]]
    
    def __init__(self, node_specs: list[dict], adjacency_list: list[list[int]]) -> None:
        self.nodes: list[Node] = [get_node(spec) for spec in node_specs]
        self.adjacency_list = adjacency_list

# %%
# create a synthetic problem

