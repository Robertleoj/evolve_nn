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
from webbrowser import Opera
import torch
import math
from typing import Any
from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
import torch.nn as nn
from uuid import uuid4
from graphviz import Digraph
from IPython.display import SVG, display
from project.graph.nodes import DataNode, InputNode, Node, OperatorNode, OutputNode, ParameterNode, node_from_spec
from project.utils.graph_utils import topsort_edge_list
import networkx as nx


@dataclass
class Graph:
    """A computational graph.

    Attributes:
        nodes: List of nodes in the graph.
        id_to_node: Mapping from node ID to node.
        adj_list: Adjacency list of the graph.
        rev_adj_list: Reversed adjacency list of the graph.
    """

    id_to_node: dict[str, Node]
    rev_adj_list: dict[str, list[str]]

    def add_node(self, node: Node) -> str:
        node_id = str(uuid4())
        self.id_to_node[node_id] = node
        return node_id

    def input_nodes(self) -> list[str]:
        return [
            node_id for node_id, node in self.id_to_node.items()
            if isinstance(self.id_to_node[node_id], InputNode)
        ]

    def operator_nodes(self) -> list[str]:
        return [
            node_id for node_id, node in self.id_to_node.items()
            if isinstance(node, OperatorNode)
        ]

    def parameter_nodes(self) -> list[str]:
        return [
            node_id for node_id, node in self.id_to_node.items()
            if isinstance(node, ParameterNode)
        ]

    def output_nodes(self) -> list[str]:
        return [
            node_id for node_id, node in self.id_to_node.items()
            if isinstance(node, OutputNode)
        ]

    def data_nodes(self) -> list[str]:
        return [
            node_id for node_id, node in self.id_to_node.items()
            if isinstance(node, DataNode)
        ]

    def get_nx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.id_to_node.keys())
        for node_id, neighbors in self.rev_adj_list.items():
            for neighbor in neighbors:
                graph.add_edge(neighbor, node_id)
        return graph


def make_graph(
    node_specs: list[dict[str, Any]],
    rev_adj_list: list[tuple[int, int]]
) -> Graph:
    nodes = [node_from_spec(spec) for spec in node_specs]
    node_ids = [str(uuid4()) for _ in nodes]

    id_rev_adj_list = defaultdict(list)
    for node_idx, backward_neighbors in enumerate(rev_adj_list):
        node_id = node_ids[node_idx]
        for neighbor_idx in backward_neighbors:
            neighbor_id = node_ids[neighbor_idx]
            id_rev_adj_list[node_id].append(neighbor_id)

    return Graph(
        id_to_node=dict(zip(node_ids, nodes)),
        rev_adj_list=id_rev_adj_list,
    )

def show_graph(graph: Graph) -> None:
    """Show the graph using Graphviz."""
    dot = Digraph()

    for node_id, node in graph.id_to_node.items():
        label = node.name
        if isinstance(node, DataNode):
            dot.node(node_id, label=label)
        elif isinstance(node, OperatorNode):
            dot.node(node_id, label=label, shape="box")

    for node_id, backward_neighbors in graph.rev_adj_list.items():
        for i, neighbor_id in enumerate(backward_neighbors):
            dot.edge(neighbor_id, node_id, label=str(i))

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))

def show_multiple_graphs(graphs: list[Graph]) -> None:
    """Show multiple graphs in a single figure, each graph labeled by its index."""
    dot = Digraph()
    dot.attr(compound='true')

    for index, graph in enumerate(graphs):
        with dot.subgraph(name=f'cluster_{index}') as c:
            c.attr(label=f'Graph {index}')
            for node_id, node in graph.id_to_node.items():
                label = node.name
                name = f"{node_id}_{index}"
                if isinstance(node, DataNode):
                    c.node(name, label=label)
                elif isinstance(node, OperatorNode):
                    c.node(name, label=label, shape="box")

            for node_id, backward_neighbors in graph.rev_adj_list.items():
                for i, back_neighbor_id in enumerate(backward_neighbors):
                    source = f"{back_neighbor_id}_{index}"
                    target = f"{node_id}_{index}"
                    c.edge(source, target, label=str(i))

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

    nodes: list[Node]
    rev_adjacency_list: list[list[int]]
    input_nodes: list[int]
    output_nodes: list[int]
    _modules: nn.ModuleDict
    _parameters: nn.ParameterDict
    

    def __init__(
        self,
        nodes: list[Node],
        rev_adjacency_list: list[list[int]],
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
        edge_list = []
        for to_idx, from_indices in enumerate(rev_adjacency_list):
            for from_idx in from_indices:
                edge_list.append((from_idx, to_idx))

        topsorted = topsort_edge_list(len(nodes), edge_list)

        topsorted_idx = {
            node_id: i for i, node_id in enumerate(topsorted)
        }

        sorted_nodes = [nodes[i] for i in topsorted]
        self.nodes = sorted_nodes

        sorted_rev_adj_list: list[list[int]] = [[] for _ in range(len(nodes))]
        for to_idx, from_indices in enumerate(rev_adjacency_list):
            sorted_to_idx = topsorted_idx[to_idx]
            sorted_back_list = []
            for from_idx in from_indices:
                sorted_from_idx = topsorted_idx[from_idx]
                sorted_back_list.append(sorted_from_idx)
            sorted_rev_adj_list[sorted_to_idx] = sorted_back_list
        
        self.rev_adjacency_list = sorted_rev_adj_list

        self._modules = nn.ModuleDict()
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, OperatorNode):
                self._modules[str(node_id)] = node.get_op()

        self._parameters = nn.ParameterDict()
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, ParameterNode):
                self._parameters[str(node_id)] = nn.Parameter(torch.empty(1))

        self.reset_parameters()

        self.input_nodes = [i for i, node in enumerate(self.nodes) if isinstance(node, InputNode)]
        self.output_nodes = [i for i, node in enumerate(self.nodes) if isinstance(node, OutputNode)]

    def reset_parameters(self) -> None:
        """Reset the parameters of the graph."""
        for param in self._parameters.values():
            nn.init.normal_(param, mean=0, std=1 / math.sqrt(2))

    @classmethod
    def from_graph(cls, graph: Graph) -> "CompiledGraph":
        """Create a compiled graph from a graph.

        Args:
            graph: The graph to compile.

        Returns:
            CompiledGraph: The compiled graph.
        """

        node_id_to_node_idx = {}
        nodes = []

        for i, (node_id, node) in enumerate(graph.id_to_node.items()):
            node_id_to_node_idx[node_id] = i
            nodes.append(node)

        idx_rev_edge_list: list[list[int]] = [[] for _ in range(len(graph.id_to_node))]

        for to_node_id, from_node_ids in graph.rev_adj_list.items():
            to_idx = node_id_to_node_idx[to_node_id]
            for from_node_id in from_node_ids:
                from_idx = node_id_to_node_idx[from_node_id]
                idx_rev_edge_list[to_idx].append(from_idx)

        return cls(
            nodes=nodes,
            rev_adjacency_list=idx_rev_edge_list,
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform inference on the compiled graph.

        Args:
            inputs: List of input tensors.

        Returns:
            List of output tensors.
        """
        data: list[None | torch.Tensor] = [None for _ in range(len(self.nodes))]

        for node_id, input in zip(self.input_nodes, inputs):
            data[node_id] = input

        for node_id, param in self._parameters.items():
            data[int(node_id)] = param

        for node_id in range(len(self.nodes)):
            self._infer_node(node_id, data)

        return [data[node_id] for node_id in self.output_nodes]

    def _infer_node(self, node_id: int, data: list[None | torch.Tensor]) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        node = self.nodes[node_id]

        if data[node_id] is not None:
            return

        if isinstance(node, OutputNode):
            input_node_ids = self.rev_adjacency_list[node_id]

            assert len(input_node_ids) == 1, "Output nodes should only have one incoming edge."
            input_node_id = input_node_ids[0]
            data[node_id] = data[input_node_id]
            return

        if isinstance(node, OperatorNode):
            input_data = []
            for i in self.rev_adjacency_list[node_id]:
                inp_data = data[i]
                assert inp_data is not None, f"Inferring node {node_id}: node {i} has not been inferred yet."
                input_data.append(inp_data)

            op = self._modules[str(node_id)]
            data[node_id] = op(input_data)


def show_compiled(graph: CompiledGraph) -> None:
    """Show the graph using Graphviz."""
    dot = Digraph()

    for node_idx, node in enumerate(graph.nodes):
        label = node.name
        if isinstance(node, DataNode):
            dot.node(str(node_idx), label=label)
        elif isinstance(node, OperatorNode):
            dot.node(str(node_idx), label=label, shape="box")

    for to_idx, from_indices in enumerate(graph.rev_adjacency_list):
        for from_idx in from_indices:
            edge = (from_idx, to_idx)
            dot.edge(
                str(from_idx), 
                str(to_idx), 
            )

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))



