"""Graph module for the project.

TODO: check that:
* graph is Connected
* graph is Acyclic
* shapes match
* input nodes have no incoming edges
* output nodes have no outgoing edges
* data nodes have exactly one incoming edge
* Operator nodes only point to Data nodes
* Data nodes only point to Operator nodes
"""
import math

import networkx as nx
import torch
import torch.nn as nn
from graphviz import Digraph
from IPython.display import SVG, display
from project.graph.nodes import DataNode, InputNode, Node, OperatorNode, OutputNode, ParameterNode, node_from_spec


def topsort(adj_list: list[list[int]]) -> list[int]:
    """Topologically sort a graph given its adjacency list.

    Args:
        adj_list: Adjacency list of the graph.

    Returns:
        List of node indices in topological order.
    """
    return list(nx.topological_sort(nx.DiGraph(adj_list)))


def reverse_adjacency_list(adj_list: list[list[int]]) -> list[list[int]]:
    """Reverse the adjacency list of a graph.

    Args:
        adj_list: Adjacency list of the graph.

    Returns:
        Reversed adjacency list.
    """
    n = len(adj_list)
    rev_adj_list = [[] for _ in range(n)]
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            rev_adj_list[j].append(i)
    return rev_adj_list


class Graph:
    """A computational graph.

    Attributes:
        nodes: List of nodes in the graph.
        ordered_edge_list: List of edges in the graph, along with the index in the input/output list.
        topsorted: List of nodes in topological order.
        adjacency_list: Adjacency list of the graph.
        rev_adjacency_list: Reversed adjacency list of the graph.
        edge_indices: Dictionary mapping edges to their index in the input/output list.
    """

    nodes: list[Node]
    ordered_edge_list: list[tuple[tuple[int, int], int]]
    topsorted: list[int]
    adjacency_list: list[list[int]]
    rev_adjacency_list: list[list[int]]
    edge_indices: dict[tuple[int, int], int]

    def __init__(
        self,
        node_specs: list[dict],
        ordered_edge_list: list[tuple[tuple[int, int], int]],
        topsorted: list[int] | None = None,
        rev_adj_list: list[list[int]] | None = None,
    ) -> None:
        """Make a graph from a list of node specs and an ordered edge list.

        Args:
            node_specs: List of node specs.
            ordered_edge_list: List of edges in the graph, along with the index in the input/output list.
            topsorted: List of nodes in topological order. If none, it will be computed.
            rev_adj_list: Reversed adjacency list of the graph. If none, it will be computed.
        """
        self.nodes: list[Node] = [node_from_spec(spec) for spec in node_specs]
        self.ordered_edge_list = ordered_edge_list

        self.topsorted = topsorted
        if topsorted is None:
            self.topsorted = topsort([edge[0] for edge in ordered_edge_list])

        self.adjacency_list = [
            [edge[0][1] for edge in ordered_edge_list if edge[0][0] == i] for i in range(len(node_specs))
        ]
        self.rev_adjacency_list = rev_adj_list
        if rev_adj_list is None:
            self.rev_adjacency_list = reverse_adjacency_list(self.adjacency_list)

        self.edge_indices = {edge[0]: edge[1] for edge in ordered_edge_list}

    def _get_label(self, node: Node) -> str:
        if isinstance(node, DataNode):
            shape_info = str(node.shape)
        elif isinstance(node, OperatorNode):
            input_shape_info = " x ".join(str(shape) for shape in node.input_shapes)
            output_shape_info = " x ".join(str(shape) for shape in node.output_shapes)
            shape_info = f"{input_shape_info} -> {output_shape_info}"

        return f"{node.name}\n{shape_info}"

    def show(self) -> None:
        """Show the graph using Graphviz."""
        dot = Digraph()

        for i, node in enumerate(self.nodes):
            label = self._get_label(node)
            if isinstance(node, DataNode):
                dot.node(str(i), label=label)
            elif isinstance(node, OperatorNode):
                dot.node(str(i), label=label, shape="box")

        for edge in self.ordered_edge_list:
            dot.edge(str(edge[0][0]), str(edge[0][1]), label=str(edge[1]))

        svg = dot.pipe(format="svg").decode("utf-8")
        display(SVG(svg))


class CompiledGraph(nn.Module):
    """Compiled graph for inference and training.

    Attributes:
        nodes: List of nodes in the graph.
        topsorted: List of nodes in topological order.
        input_nodes: List of input node indices.
        output_nodes: List of output node indices.
        rev_adjacency_list: Reversed adjacency list of the graph.
        edge_indices: Dictionary mapping edges to their index in the input/output list.
    """

    nodes: nn.ModuleList
    topsorted: list[int]
    input_nodes: list[int]
    output_nodes: list[int]
    rev_adjacency_list: list[list[int]]
    edge_indices: dict[tuple[int, int], int]

    def __init__(
        self,
        nodes: list[Node],
        topsorted: list[int],
        input_nodes: list[int],
        output_nodes: list[int],
        rev_adjacency_list: list[list[int]],
        edge_indices: dict[tuple[int, int] : int],
    ) -> None:
        """Create a compiled graph.

        Args:
            nodes: List of nodes in the graph.
            topsorted: List of nodes in topological order.
            input_nodes: List of input node indices.
            output_nodes: List of output node indices.
            rev_adjacency_list: Reversed adjacency list of the graph.
            edge_indices: Dictionary mapping edges to their index in the input/output list.
        """
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.topsorted = topsorted
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.rev_adjacency_list = rev_adjacency_list
        self.edge_indices = edge_indices

        self._initialize_parameters()

    def _initialize_parameters(self):
        self._parameters = nn.ParameterDict()
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, ParameterNode):
                self._parameters[str(node_id)] = nn.Parameter(torch.empty(node.shape))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the graph."""
        for param in self._parameters.values():
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    @classmethod
    def from_graph(cls, graph: Graph) -> "CompiledGraph":
        """Create a compiled graph from a graph.

        Args:
            graph: The graph to compile.

        Returns:
            CompiledGraph: The compiled graph.
        """
        input_nodes = [i for i, node in enumerate(graph.nodes) if isinstance(node, InputNode)]

        output_nodes = [i for i, node in enumerate(graph.nodes) if isinstance(node, OutputNode)]

        return cls(
            graph.nodes, graph.topsorted, input_nodes, output_nodes, graph.rev_adjacency_list, graph.edge_indices
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform inference on the compiled graph.

        Args:
            inputs: List of input tensors.

        Returns:
            List of output tensors.
        """
        data = [None for _ in range(len(self.nodes))]

        for node_id, input in zip(self.input_nodes, inputs):
            data[node_id] = input

        for node_id, param in self._parameters.items():
            data[int(node_id)] = param

        for node_idx in self.topsorted:
            if data[node_idx] is not None:
                continue

            self._infer_node(node_idx, data)

        return [data[node_idx] for node_idx in self.output_nodes]

    def _infer_data_node(self, node_idx: int, data: list[torch.Tensor | list[torch.Tensor]]) -> None:
        """Infer the value of a data node.

        Args:
            node_idx: Index of the data node.
            data: List of tensors for each node.
        """
        # get data from output node
        output_node_idx = self.rev_adjacency_list[node_idx][0]
        output_idx = self.edge_indices[(output_node_idx, node_idx)]

        data[node_idx] = data[output_node_idx][output_idx]

    def _infer_node(self, node_idx: int, data: list[torch.Tensor | list[torch.Tensor]]) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        node = self.nodes[node_idx]
        if isinstance(node, DataNode):
            self._infer_data_node(node_idx, data)
            return

        input_node_indices = self.rev_adjacency_list[node_idx]
        input_data = [None for _ in input_node_indices]

        for input_node_idx in input_node_indices:
            edge = (input_node_idx, node_idx)
            input_index = self.edge_indices[edge]
            input_data[input_index] = data[input_node_idx]

        data[node_idx] = node(input_data)
