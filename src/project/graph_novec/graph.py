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
import torch
from dataclasses import dataclass
from copy import deepcopy
import torch.nn as nn
from uuid import uuid4
from graphviz import Digraph
from IPython.display import SVG, display
from project.graph_novec.nodes import DataNode, InputNode, Node, OperatorNode, OutputNode, ParameterNode, node_from_name
import networkx as nx

def topsort_adj_list(adj_list: list[list[int]]) -> list[int]:
    """Topologically sort a graph given its adjacency list.

    Args:
        adj_list: Adjacency list of the graph.

    Returns:
        List of node indices in topological order.
    """
    return list(nx.topological_sort(nx.DiGraph(adj_list)))

def topsort_edge_list(num_nodes: int, edge_list: list[tuple[int, int]]) -> list[int]:
    """Topologically sort a graph given its edge list.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_list: List of edges in the graph.

    Returns:
        List of node indices in topological order.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_list)
    return list(nx.topological_sort(graph))


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
    node_ids: list[str]
    id_to_node: dict[str, Node]
    edge_list: list[tuple[str, str]]

    def add_node(self, node: Node) -> str:
        self.nodes.append(node)
        node_id = str(uuid4())
        self.node_ids.append(node_id)
        self.id_to_node[node_id] = node
        return node_id

    def operator_nodes(self) -> list[str]:
        return [
            node_id for node_id in self.node_ids 
            if isinstance(self.id_to_node[node_id], OperatorNode)
        ]

    def output_nodes(self) -> list[str]:
        return [
            node_id for node_id in self.node_ids 
            if isinstance(self.id_to_node[node_id], OutputNode)
        ]

    def remove_node(self, node_id) -> None:
        node_idx = self.node_ids.index(node_id)
        self.nodes.pop(node_idx)
        self.node_ids.pop(node_idx)
        del self.id_to_node[node_id]

    def adjacency_list(self, reverse: bool = False) -> dict[str, list[str]]:
        adj_list = {node_id: [] for node_id in self.node_ids}
        for (a, b) in self.edge_list:
            if reverse:
                b, a = a, b
            adj_list[a].append(b)

        return adj_list

    def get_nx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.node_ids)
        graph.add_edges_from(self.edge_list)
        return graph

    def reset_edge_list(self, edge_list: list[str, str]) -> None:
        self.edge_list = edge_list


def make_graph(
    node_names: list[str],
    edge_list: list[tuple[int, int]],
) -> Graph:
    nodes = [node_from_name(name) for name in node_names]
    node_ids = [str(uuid4()) for _ in nodes]
    edge_list_uuid = [
        (node_ids[edge[0]], node_ids[edge[1]]) for edge in edge_list
    ]
    id_to_node = dict(zip(node_ids, nodes))

    return Graph(nodes, node_ids, id_to_node, edge_list_uuid)

def show_graph(graph: Graph) -> None:
    """Show the graph using Graphviz."""
    dot = Digraph()

    for node_id, node in zip(graph.node_ids, graph.nodes):
        label = node.name
        if isinstance(node, DataNode):
            dot.node(node_id, label=label)
        elif isinstance(node, OperatorNode):
            dot.node(node_id, label=label, shape="box")

    for edge in graph.edge_list:
        dot.edge(
            str(edge[0]), 
            str(edge[1]), 
        )

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))


