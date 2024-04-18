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
from collections import defaultdict
from typing import Any
from uuid import uuid4

import networkx as nx
import torch
import torch.nn as nn
from graphviz import Digraph
from IPython.display import SVG, display
from project.utils.graph_utils import topsort_edge_list


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
    """A node that represents an operation.

    Attributes:
        input_shapes: Shapes of the input tensors.
        output_shapes: Shapes of the output tensors.
    """

    n_inputs: tuple[int, int | None]
    inputs_ordered: bool = False

    def get_op(self) -> nn.Module:
        """Perform the operation."""
        raise NotImplementedError


class AddMod(nn.Module):
    def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return sum(inp)  # type: ignore


class AddNode(OperatorNode):
    """A node that adds two or more tensors."""

    name = "add"
    n_inputs = (1, None)

    def get_op(self) -> nn.Module:
        """Perform the addition operation."""
        return AddMod()


class ProdMod(nn.Module):
    def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return math.prod(inp)  # type: ignore


class ProdNode(OperatorNode):
    """A node that multiplies two or more tensors."""

    name = "prod"
    n_inputs = (1, None)

    def get_op(self) -> nn.Module:
        """Perform the multiplication operation."""
        return ProdMod()


class GELUMod(nn.Module):
    def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return nn.functional.gelu(inp[0])


class GELUNode(OperatorNode):
    """A node that applies the ReLu function."""

    name = "GELU"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the ReLu operation."""
        return GELUMod()


class LogMod(nn.Module):
    def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return torch.log(inp[0])


class LogNode(OperatorNode):
    """A node that applies the log function."""

    name = "log"

    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the log operation."""
        return LogMod()


class ExpMod(nn.Module):
    def forward(self, inp: list[torch.Tensor]) -> torch.Tensor:
        return torch.exp(inp[0])


class ExpNode(OperatorNode):
    """A node that applies the exp function."""

    name = "exp"
    n_inputs = (1, 1)

    def get_op(self) -> nn.Module:
        """Perform the exp operation."""
        return ExpMod()


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
    ordered_input_nodes: list[str]
    ordered_output_nodes: list[str]
    subgraphs: list["Graph"]
    is_subgraph: bool = False
    ordered_respone_input_nodes: list[str] | None
    loss_output_node: str | None

    def __init__(
        self,
        *,
        id_to_node: dict[str, Node],
        rev_adj_list: dict[str, list[str]],
        ordered_input_nodes: list[str],
        ordered_output_nodes: list[str],
        subgraphs: list["Graph"] = [],
        ordered_respoone_input_nodes: list[str] = None,
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
            if isinstance(node, LossOutputNode):
                self.loss_output_node = node_id
            if node_id not in self.rev_adj_list:
                self.rev_adj_list[node_id] = []
        self.ordered_input_nodes = ordered_input_nodes
        self.ordered_output_nodes = ordered_output_nodes
        self.ordered_response_input_nodes = ordered_respoone_input_nodes
        self.subgraphs = subgraphs

    def add_node(self, node: Node) -> str:
        assert isinstance(node, Node)
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
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(self.id_to_node[node_id], InputNode)]

    def operator_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, OperatorNode)]

    def parameter_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, ParameterNode)]

    def output_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, OutputNode)]

    def data_nodes(self) -> list[str]:
        return [node_id for node_id, node in self.id_to_node.items() if isinstance(node, DataNode)]

    def get_nx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.id_to_node.keys())
        for node_id, neighbors in self.rev_adj_list.items():
            for neighbor in neighbors:
                graph.add_edge(neighbor, node_id)
        return graph


class SubGraphNode(OperatorNode):
    """A node that represents a subgraph."""

    name = "graph"
    subgraph: Graph

    def __init__(self, subgraph: Graph) -> None:
        num_inputs = len(subgraph.ordered_input_nodes)
        self.subgraph = subgraph
        self.n_inputs = (num_inputs, num_inputs)

    def get_op(self) -> "CompiledGraph":
        return SubCompiledGraph.from_graph(self.subgraph)


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

    nodes = []
    for spec in node_specs:
        if spec["name"] == "graph":
            subgraph_idx = spec["subgraph_idx"]
            node = SubGraphNode(subgraph=subgraphs[subgraph_idx])
            nodes.append(node)
        else:
            nodes.append(node_from_spec(spec))

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
            if isinstance(node, OutputNode):
                first_output_node = i
                break
        output_node_order = [first_output_node]

    output_node_id_order = [node_ids[i] for i in output_node_order]

    return Graph(
        id_to_node=dict(zip(node_ids, nodes)),
        rev_adj_list=id_rev_adj_list,
        ordered_input_nodes=input_node_id_order,
        ordered_output_nodes=output_node_id_order,
        subgraphs=subgraphs,
    )


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
    stored_modules: nn.ModuleDict
    stored_parameters: nn.ParameterDict
    curr_data: list[None | torch.Tensor]

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

        topsorted_idx = {node_id: i for i, node_id in enumerate(topsorted)}

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

        self.stored_modules = nn.ModuleDict()
        self.stored_parameters = nn.ParameterDict()
        self.input_nodes = []
        self.output_nodes = []
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, OperatorNode):
                self.stored_modules[str(node_id)] = node.get_op()

            if isinstance(node, ParameterNode):
                self.stored_parameters[str(node_id)] = nn.Parameter(torch.empty(1))

            if isinstance(node, InputNode):
                self.input_nodes.append(node_id)

            if isinstance(node, OutputNode):
                self.output_nodes.append(node_id)

        self.reset_parameters()
        self.reset_data()

    def reset_parameters(self) -> None:
        """Reset the parameters of the graph."""
        for param in self.stored_parameters.values():
            nn.init.normal_(param, mean=0, std=1 / math.sqrt(2))

    def reset_data(self) -> None:
        """Reset the data of the graph."""
        self.curr_data = [None for _ in self.nodes]

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
        self.reset_data()

        for node_id, input in zip(self.input_nodes, inputs):
            self.curr_data[node_id] = input

        for node_id, param in self.stored_parameters.items():
            self.curr_data[int(node_id)] = param

        for node_id, node in enumerate(self.nodes):
            if isinstance(node, ResponseInputNode | LossOutputNode):
                break

            self.infer_node(node_id)

        return [self.curr_data[node_id] for node_id in self.output_nodes]

    def infer_node(self, node_id: int) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        node = self.nodes[node_id]

        if self.curr_data[node_id] is not None:
            return

        if isinstance(node, OutputNode):
            input_node_ids = self.rev_adjacency_list[node_id]

            assert len(input_node_ids) == 1, "Output nodes should only have one incoming edge."
            input_node_id = input_node_ids[0]
            self.curr_data[node_id] = self.curr_data[input_node_id]
            return

        if isinstance(node, OperatorNode):
            input_data = []
            for i in self.rev_adjacency_list[node_id]:
                inp_data = self.curr_data[i]
                assert inp_data is not None, f"Inferring node {node_id}: node {i} has not been inferred yet."
                input_data.append(inp_data)

            op = self.stored_modules[str(node_id)]
            self.curr_data[node_id] = op(input_data)


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
            dot.edge(
                str(from_idx),
                str(to_idx),
            )

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))


class SubCompiledGraph(CompiledGraph):
    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:  # type: ignore
        outputs = super().forward(inputs)
        return outputs[0]


def make_recursive_graph(graph: Graph, dot: Digraph | None = None, show_node_ids: bool = False) -> Digraph:
    if dot is None:
        dot = Digraph()

    for node_id, node in graph.id_to_node.items():
        name = node.name
        if show_node_ids:
            name = f"{name} ({node_id})"

        label = name
        if isinstance(node, DataNode):
            if isinstance(node, InputNode):
                input_idx = next(
                    i for i, iter_node_id in enumerate(graph.ordered_input_nodes) if node_id == iter_node_id
                )

                label = f"{name} {input_idx}"
            dot.node(node_id, label=label)
        elif isinstance(node, SubGraphNode):
            subgraph_idx = None
            for i, subgraph in enumerate(graph.subgraphs):
                if subgraph is node.subgraph:
                    subgraph_idx = i
                    break
            assert subgraph_idx is not None, "Subgraph not found."
            label = f"{name} {subgraph_idx}"
            dot.node(node_id, label=label, shape="box")

        elif isinstance(node, OperatorNode):
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


op_nodes: list[type[OperatorNode]] = [AddNode, ProdNode, GELUNode, ExpNode, SubGraphNode]

data_nodes: list[type[DataNode]] = [
    InputNode,
    OutputNode,
    ParameterNode,
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
