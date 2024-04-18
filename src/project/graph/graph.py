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
from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import uuid4

import networkx as nx
import project.graph.nodes as nodes_
from graphviz import Digraph
from IPython.display import SVG, display


class Graph:
    """A computational graph.

    Attributes:
        nodes: List of nodes in the graph.
        id_to_node: Mapping from node ID to node.
        adj_list: Adjacency list of the graph.
        rev_adj_list: Reversed adjacency list of the graph.
    """

    id_to_node: dict[str, nodes_.Node]
    rev_adj_list: dict[str, list[str]]
    ordered_input_nodes: list[str]
    ordered_output_nodes: list[str]
    subgraphs: list["Graph"]
    is_subgraph: bool = False
    ordered_respone_input_nodes: list[str] | None
    loss_output_node: str | None

    def __init__(
        self,
        *,
        id_to_node: dict[str, nodes_.Node],
        rev_adj_list: dict[str, list[str]],
        ordered_input_nodes: list[str],
        ordered_output_nodes: list[str],
        subgraphs: list["Graph"] = [],
        ordered_respoone_input_nodes: list[str] | None = None,
    ) -> None:
        """Create a graph.

        Args:
            nodes: List of nodes in the graph.
            adj_list: Adjacency list of the graph.
            rev_adj_list: Reversed adjacency list of the graph.
        """
        self.id_to_node = id_to_node
        self.rev_adj_list = dict(rev_adj_list)

        self.loss_output_node = None
        for node_id, node in self.id_to_node.items():
            if isinstance(node, nodes_.LossOutputNode):
                self.loss_output_node = node_id
            if node_id not in self.rev_adj_list:
                self.rev_adj_list[node_id] = []
        self.ordered_input_nodes = ordered_input_nodes
        self.ordered_output_nodes = ordered_output_nodes
        self.ordered_response_input_nodes = ordered_respoone_input_nodes
        self.subgraphs = subgraphs

    def add_node(self, node: nodes_.Node) -> str:
        assert isinstance(node, nodes_.Node)
        node_id = str(uuid4())
        self.id_to_node[node_id] = node
        self.rev_adj_list[node_id] = []
        return node_id

    def delete_node(self, node_id) -> str:
        del self.id_to_node[node_id]
        for neighbors in self.rev_adj_list.values():
            while node_id in neighbors:
                neighbors.remove(node_id)
        del self.rev_adj_list[node_id]

        return node_id

    def delete_edge(self, from_node_id: str, to_node_id: str) -> None:
        self.rev_adj_list[to_node_id].remove(from_node_id)

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        self.rev_adj_list[to_node_id].append(from_node_id)

    def add_edges(self, edges: list[tuple[str, str]]) -> None:
        for from_node_id, to_node_id in edges:
            self.add_edge(from_node_id, to_node_id)

    @property
    def adj_list(self) -> dict[str, list[str]]:
        adj_list = defaultdict(list)
        for node_id, neighbors in self.rev_adj_list.items():
            for neighbor in neighbors:
                adj_list[neighbor].append(node_id)
        for node_id in self.id_to_node.keys():
            if node_id not in adj_list:
                adj_list[node_id] = []
        return dict(adj_list)

    @property
    def edge_list(self) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for node_id, neighbors in self.rev_adj_list.items():
            for neighbor in neighbors:
                edges.append((neighbor, node_id))
        return edges

    def input_nodes(self) -> list[str]:
        return [
            node_id
            for node_id, node in self.id_to_node.items()
            if isinstance(self.id_to_node[node_id], nodes_.InputNode)
        ]

    def operator_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, nodes_.OperatorNode)]

    def parameter_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, nodes_.ParameterNode)]

    def output_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, nodes_.OutputNode)]

    def data_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, nodes_.DataNode)]

    def get_nx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.id_to_node.keys())
        for node_id, neighbors in self.rev_adj_list.items():
            for neighbor in neighbors:
                graph.add_edge(neighbor, node_id)
        return graph


def make_graph(
    node_specs: list[dict[str, Any]],
    rev_adj_list: list[list[int]],
    input_node_order: list[int],
    subgraph_specs: list[dict[str, Any]] | None = None,
    output_node_order: list[int] | None = None,
) -> Graph:
    subgraphs = []
    if subgraph_specs is not None:
        for spec in subgraph_specs:
            subgraphs.append(make_graph(**spec))

    nodes: list[nodes_.Node] = []
    node: nodes_.Node

    for spec in node_specs:
        if spec["name"] == "graph":
            subgraph_idx = spec["subgraph_idx"]
            node = nodes_.SubGraphNode(subgraph=subgraphs[subgraph_idx])
            nodes.append(node)
        else:
            nodes.append(nodes_.node_from_spec(spec))

    node_ids = [str(uuid4()) for _ in nodes]

    id_rev_adj_list = defaultdict(list)
    for node_idx, backward_neighbors in enumerate(rev_adj_list):
        node_id = node_ids[node_idx]
        for neighbor_idx in backward_neighbors:
            neighbor_id = node_ids[neighbor_idx]
            id_rev_adj_list[node_id].append(neighbor_id)

    input_node_id_order = [node_ids[i] for i in input_node_order]
    if output_node_order is None:
        first_output_node = None
        for i, node in enumerate(nodes):
            if isinstance(node, nodes_.OutputNode):
                first_output_node = i
                break
        assert first_output_node is not None, "No output node found."
        output_node_order = [first_output_node]

    output_node_id_order = [node_ids[i] for i in output_node_order]

    return Graph(
        id_to_node=dict(zip(node_ids, nodes)),
        rev_adj_list=id_rev_adj_list,
        ordered_input_nodes=input_node_id_order,
        ordered_output_nodes=output_node_id_order,
        subgraphs=subgraphs,
    )


def make_recursive_graph(graph: Graph, dot: Digraph | None = None, show_node_ids: bool = False) -> Digraph:
    if dot is None:
        dot = Digraph()

    for node_id, node in graph.id_to_node.items():
        name = node.name
        if show_node_ids:
            name = f"{name} ({node_id})"

        label = name
        if isinstance(node, nodes_.DataNode):
            if isinstance(node, nodes_.InputNode):
                input_idx = next(
                    i for i, iter_node_id in enumerate(graph.ordered_input_nodes) if node_id == iter_node_id
                )

                label = f"{name} {input_idx}"
            dot.node(node_id, label=label)
        elif isinstance(node, nodes_.SubGraphNode):
            subgraph_idx = None
            for i, subgraph in enumerate(graph.subgraphs):
                if subgraph is node.subgraph:
                    subgraph_idx = i
                    break
            assert subgraph_idx is not None, "Subgraph not found."
            label = f"{name} {subgraph_idx}"
            dot.node(node_id, label=label, shape="box")

        elif isinstance(node, nodes_.OperatorNode):
            dot.node(node_id, label=label, shape="box")

    for node_id, backward_neighbors in graph.rev_adj_list.items():
        for i, neighbor_id in enumerate(backward_neighbors):
            dot.edge(neighbor_id, node_id, label=str(i))

    for i, subgraph in enumerate(graph.subgraphs):
        with dot.subgraph(name="cluster_" + str(uuid4())) as sg:
            make_recursive_graph(subgraph, dot=sg, show_node_ids=show_node_ids)
            label = f"Graph {i}"
            sg.attr(label=label)

    return dot


def get_graph_svg(graph: Graph, show_node_ids: bool = False) -> SVG:
    dot = make_recursive_graph(graph, show_node_ids=show_node_ids)
    svg = dot.pipe(format="svg").decode("utf-8")

    return SVG(svg)


def show_graph(graph: Graph, show_node_ids: bool = False) -> None:
    """Show the graph using Graphviz."""
    display(get_graph_svg(graph, show_node_ids=show_node_ids))


def show_multiple_graphs(graphs: list[Graph]) -> None:
    """Show multiple graphs in a single figure, each graph labeled by its index."""
    dot = Digraph()
    dot.attr(compound="true")

    for index, graph in enumerate(graphs):
        with dot.subgraph(name=f"cluster_{index}") as c:
            c.attr(label=f"Graph {index}")
            for node_id, node in graph.id_to_node.items():
                label = node.name
                name = f"{node_id}_{index}"
                if isinstance(node, nodes_.DataNode):
                    c.node(name, label=label)
                elif isinstance(node, nodes_.OperatorNode):
                    c.node(name, label=label, shape="box")

            for node_id, backward_neighbors in graph.rev_adj_list.items():
                for i, back_neighbor_id in enumerate(backward_neighbors):
                    source = f"{back_neighbor_id}_{index}"
                    target = f"{node_id}_{index}"
                    c.edge(source, target, label=str(i))

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))
