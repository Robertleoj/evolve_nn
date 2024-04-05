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
import networkx as nx


# %%
class Node(nn.Module):
    name: str

    def get_spec(self) -> dict:
        raise NotImplementedError

    def from_spec(self, spec: dict) -> None:
        raise NotImplementedError



# %%
# Data nodes
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

class ParameterNode(DataNode):
    name = "parameter"


class HiddenNode(DataNode):
    name = "hidden"


class OutputNode(DataNode):
    name = "output"



# %%
# operator nodes

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
    node_type.name: node_type
    for node_type in [
        InputNode,
        HiddenNode,
        OutputNode,
        ParameterNode,
        AddNode,
        MatmulNode,
        ReLuNode
    ]
}
node_map


# %%
def get_node(spec: dict): 
    return node_map[spec["name"]].from_spec(spec)


# %%
add_node = get_node({
    "name": "add",
    "input_shapes": [(1, 2), (1, 2)],
    "output_shapes": [(1, 2)]
})

# %%
add_node([torch.tensor([1, 2]), torch.tensor([3, 4])])


# %%
def topsort(adj_list: list[list[int]]) -> list[int]:
    return list(nx.topological_sort(nx.DiGraph(adj_list)))

def reverse_adjacency_list(adj_list: list[list[int]]) -> list[list[int]]:
    n = len(adj_list)
    rev_adj_list = [[] for _ in range(n)]
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            rev_adj_list[j].append(i)
    return rev_adj_list


# %%
class Graph:
    nodes: list[Node]
    adjacency_list: list[list[int]]
    topsorted: list[int]
    rev_adjacency_list: list[list[int]]
    
    def __init__(
        self, 
        node_specs: list[dict], 
        ordered_edge_list: list[tuple[tuple[int, int], int]], 
        topsorted: list[int] | None = None,
        rev_adj_list: list[list[int]] | None = None
    ) -> None:
        """Make a graph from a list of node specs and an ordered edge list.

        Args:
            node_specs: List of node specs.
            ordered_edge_list: List of edges in the graph, along with the index in the input/output list.
            topsorted: List of nodes in topological order. Defaults to None.
        """
        self.nodes: list[Node] = [get_node(spec) for spec in node_specs]
        self.ordered_edge_list = ordered_edge_list

        self.topsorted = topsorted
        if topsorted is None:
            self.topsorted = topsort([edge[0] for edge in ordered_edge_list])

        self.adjacency_list = [
            [edge[0][1] for edge in ordered_edge_list if edge[0][0] == i]
            for i in range(len(node_specs))
        ]
        self.rev_adjacency_list = rev_adj_list
        if rev_adj_list is None:
            self.rev_adjacency_list = reverse_adjacency_list(self.adjacency_list)

        self.edge_indices = {
            edge[0]: edge[1]
            for edge in ordered_edge_list
        }

    def __repr__(self) -> str:
        return f"Graph(\n\t{self.nodes}, {self.ordered_edge_list})"



# %%
# create a synthetic problem: add two numbers and apply ReLu
sample_graph_spec = {
    "node_specs" : [
        {
            "name": "input",
            "shape": (1, 2)
        },
        {
            "name": "input",
            "shape": (1, 2)
        },
        {
            "name": "add",
            "input_shapes": [(1, 2), (1, 2)],
            "output_shapes": [(1, 2)]
        },
        {
            "name": "hidden",
            "shape": (1, 2),
        },
        {
            "name": "relu",
            "input_shapes": [(1, 2)],
            "output_shapes": [(1, 2)]
        },
        {
            "name": "output",
            "shape": (1, 2)
        }
    ],
    "ordered_edge_list": [
        ((0, 2), 0),
        ((1, 2), 1),
        ((2, 3), 0),
        ((3, 4), 0),
        ((4, 5), 0)
    ]
}


# %%
graph = Graph(**sample_graph_spec)
graph
