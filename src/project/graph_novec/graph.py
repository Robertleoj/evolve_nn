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

    def __init__(
        self,
        node_names: list[str],
        edge_list: list[tuple[int, int]],
    ) -> None:
        """Make a graph from a list of node specs and an ordered edge list.

        Args:
            node_specs: List of node specs.
            ordered_edge_list: List of edges in the graph, along with the index in the input/output list.
            topsorted: List of nodes in topological order. If none, it will be computed.
            rev_adj_list: Reversed adjacency list of the graph. If none, it will be computed.
        """
        self.nodes = [node_from_name(name) for name in node_names]
        self.node_ids = [str(uuid4()) for _ in self.nodes]
        self.edge_list = [
            (self.node_ids[edge[0]], self.node_ids[edge[1]]) for edge in edge_list
        ]
        self.id_to_node = dict(zip(self.node_ids, self.nodes))

    def show(self) -> None:
        """Show the graph using Graphviz."""
        dot = Digraph()

        for node_id, node in zip(self.node_ids, self.nodes):
            label = node.name
            if isinstance(node, DataNode):
                dot.node(node_id, label=label)
            elif isinstance(node, OperatorNode):
                dot.node(node_id, label=label, shape="box")

        for edge in self.edge_list:
            dot.edge(
                str(edge[0]), 
                str(edge[1]), 
            )

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
    input_nodes: list[int]
    output_nodes: list[int]
    rev_adjacency_list: list[list[int]]

    def __init__(
        self,
        nodes: list[Node],
        edge_list: list[tuple[int, int]]
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
        topsorted = topsort_edge_list(len(nodes), edge_list)

        edge_list = [(topsorted[edge[0]], topsorted[edge[1]]) for edge in edge_list]

        adj_list = [[] for _ in range(len(nodes))]
        rev_adj_list = [[] for _ in range(len(nodes))]

        for edge in edge_list:
            adj_list[edge[0]].append(edge[1])
            rev_adj_list[edge[1]].append(edge[0])

        self.nodes = nn.ModuleList([nodes[i] for i in topsorted])
        self.rev_adjacency_list = reverse_adjacency_list(adj_list)

        self.input_nodes = [i for i, node in enumerate(nodes) if isinstance(node, InputNode)]
        self.output_nodes = [i for i, node in enumerate(nodes) if isinstance(node, OutputNode)]

        self._initialize_parameters()

    def _initialize_parameters(self):
        self._parameters = nn.ParameterDict()
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, ParameterNode):
                self._parameters[str(node_id)] = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the graph."""
        for param in self._parameters.values():
            nn.init.normal_(param, mean=0, std=1)

    @classmethod
    def from_graph(cls, graph: Graph) -> "CompiledGraph":
        """Create a compiled graph from a graph.

        Args:
            graph: The graph to compile.

        Returns:
            CompiledGraph: The compiled graph.
        """
        node_id_to_node_idx = {node_id: i for i, node_id in enumerate(graph.node_ids)}

        edge_list = [(node_id_to_node_idx[edge[0]], node_id_to_node_idx[edge[1]]) for edge in graph.edge_list]

        return cls(graph.nodes, edge_list)

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

        for node_id in range(len(self.nodes)):
            if data[node_id] is not None:
                continue

            self._infer_node(node_id, data)

        return [data[node_id] for node_id in self.output_nodes]

    def _infer_node(self, node_id: int, data: list[torch.Tensor]) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        node = self.nodes[node_id]

        if isinstance(node, DataNode):
            assert isinstance(node, OutputNode), "Should only have to infer output nodes."
            input_node_ids = self.rev_adjacency_list[node_id]
            assert len(input_node_ids) == 1, "Output nodes should only have one incoming edge."

            input_node_id = input_node_ids[0]
            data[node_id] = data[input_node_id]
            return

        input_data = [data[i] for i in self.rev_adjacency_list[node_id]]
        data[node_id] = node(input_data)
