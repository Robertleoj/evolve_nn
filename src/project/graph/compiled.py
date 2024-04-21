from __future__ import annotations

import math

import numpy as np
import project.foundation.graph as cpp_graph_
import project.graph.graph as graph_
import project.graph.nodes as nodes_
import project.utils.graph_utils as graph_utils_
from graphviz import Digraph
from IPython.display import SVG, display
from project.type_defs import NumpyModule


class CompiledGraph:
    """Compiled graph for inference and training.

    Attributes:
        nodes: List of nodes in the graph.
        topsorted: List of nodes in topological order.
        input_nodes: List of input node indices.
        output_nodes: List of output node indices.
        rev_adjacency_list: Reversed adjacency list of the graph.
        edge_indices: Dictionary mapping edges to their index in the input/output list.
    """

    nodes: list[nodes_.Node]
    rev_adjacency_list: list[list[int]]
    input_nodes: list[int]
    output_nodes: list[int]
    response_input_nodes: list[int]
    loss_output_node: int
    stored_modules: dict[int, NumpyModule]
    stored_parameters: dict[int, np.ndarray]
    curr_data: list[None | np.ndarray] | None
    is_subgraph: bool = False
    graph: graph_.Graph

    def __init__(
        self, nodes: list[nodes_.Node], rev_adjacency_list: list[list[int]], parent_graph: graph_.Graph
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
        self.graph = parent_graph

        edge_list = []
        for to_idx, from_indices in enumerate(rev_adjacency_list):
            for from_idx in from_indices:
                edge_list.append((from_idx, to_idx))

        topsorted = graph_utils_.topsort_edge_list(len(nodes), edge_list)

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

        self.stored_modules = {}
        self.stored_parameters = {}
        self.input_nodes = []
        self.output_nodes = []
        self.response_input_nodes = []
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, nodes_.OperatorNode):
                self.stored_modules[node_id] = node.get_op()

            if isinstance(node, nodes_.ParameterNode):
                self.stored_parameters[node_id] = np.zeros((1,))

            if isinstance(node, nodes_.InputNode):
                self.input_nodes.append(node_id)

            if isinstance(node, nodes_.OutputNode):
                self.output_nodes.append(node_id)

            if isinstance(node, nodes_.ResponseInputNode):
                self.response_input_nodes.append(node_id)

            if isinstance(node, nodes_.LossOutputNode):
                self.loss_output_node = node_id

        self.reset_parameters()
        self.nuke_data()

    def reset_parameters(self) -> None:
        """Reset the parameters of the graph."""
        for param in self.stored_parameters.values():
            param[...] = np.random.randn(*param.shape) / math.sqrt(2)

    def nuke_data(self) -> None:
        """Reset the data of the graph."""
        self.curr_data = None

    def init_data(self) -> None:
        self.curr_data = [None for _ in range(len(self.nodes))]

    @classmethod
    def from_graph(cls, graph: graph_.Graph) -> "CompiledGraph":
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

        return cls(nodes=nodes, rev_adjacency_list=idx_rev_edge_list, parent_graph=graph)

    def forward(self, inputs: list[np.array]) -> list[np.array]:
        """Perform inference on the compiled graph.

        Args:
            inputs: List of input tensors.

        Returns:
            List of output tensors.
        """
        try:
            self.init_data()
            assert self.curr_data is not None

            for node_id, input in zip(self.input_nodes, inputs):
                self.curr_data[node_id] = input

            for str_node_id, param in self.stored_parameters.items():
                self.curr_data[int(str_node_id)] = param

            for node_id, node in enumerate(self.nodes):
                if isinstance(node, nodes_.ResponseInputNode | nodes_.LossOutputNode):
                    continue

                self.infer_node(node_id)

            output: list[np.ndarray] = []
            for node_id in self.output_nodes:
                data = self.curr_data[node_id]
                assert data is not None, f"Output node {node_id} has not been inferred."

                output.append(data)
            return output
        except Exception as e:
            show_compiled(self)
            raise e

    def infer_node(self, node_id: int) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        assert self.curr_data is not None

        node = self.nodes[node_id]

        if self.curr_data[node_id] is not None:
            return

        if isinstance(node, nodes_.DataNode):
            input_node_ids = self.rev_adjacency_list[node_id]

            assert len(input_node_ids) == 1, "Output nodes should only have one incoming edge."
            input_node_id = input_node_ids[0]
            inp = self.curr_data[input_node_id]

            if inp is not None:
                self.curr_data[node_id] = inp

            return

        if isinstance(node, nodes_.OperatorNode):
            input_data = []
            for i in self.rev_adjacency_list[node_id]:
                inp_data = self.curr_data[i]
                if inp_data is None:
                    return

                input_data.append(inp_data)

            op = self.stored_modules[node_id]
            self.curr_data[node_id] = op(input_data)

    def response_forward(self, inputs: list[np.ndarray]) -> np.ndarray:
        assert not self.is_subgraph, "response_forward should only be called on the top-level graph."
        assert self.curr_data is not None, "Data has not been initialized."

        for node_id, input in zip(self.response_input_nodes, inputs):
            self.curr_data[node_id] = input

        for node_id, node in enumerate(self.nodes):
            self.infer_node(node_id)

        output = self.curr_data[self.loss_output_node]
        assert output is not None, "Loss output node has not been inferred."

        self.nuke_data()

        return output


def show_compiled(graph: CompiledGraph, use_regular: bool = True) -> None:
    """Show the graph using Graphviz."""
    if use_regular:
        graph_.show_graph(graph.graph)
        return

    dot = Digraph()

    for node_idx, node in enumerate(graph.nodes):
        label = node.name
        if isinstance(node, nodes_.DataNode):
            dot.node(node_idx, label=label)
        elif isinstance(node, nodes_.OperatorNode):
            dot.node(node_idx, label=label, shape="box")

    for to_idx, from_indices in enumerate(graph.rev_adjacency_list):
        for from_idx in from_indices:
            dot.edge(
                str(from_idx),
                str(to_idx),
            )

    svg = dot.pipe(format="svg").decode("utf-8")
    display(SVG(svg))


class SubCompiledGraph(CompiledGraph):
    is_subgraph = True

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:  # type: ignore
        outputs = super().forward(inputs)
        return outputs[0]


def to_cpp_compiled(graph: graph_.Graph) -> cpp_graph_.CompiledGraph:
    compiled_graph = CompiledGraph.from_graph(graph)
    cpp_nodes = [nodes_.to_cpp_node(node) for node in compiled_graph.nodes]

    return cpp_graph_.CompiledGraph(
        cpp_nodes,
        compiled_graph.rev_adjacency_list,
        input_order=compiled_graph.input_nodes,
        output_order=compiled_graph.output_nodes,
    )
