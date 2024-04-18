from __future__ import annotations
import project.graph.nodes as nodes_
import torch
import torch.nn as nn
import project.utils.graph_utils as graph_utils_
from graphviz import Digraph
from IPython.display import display, SVG
import project.graph.graph as graph_
import math

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

    nodes: list[nodes_.Node]
    rev_adjacency_list: list[list[int]]
    input_nodes: list[int]
    output_nodes: list[int]
    stored_modules: nn.ModuleDict
    stored_parameters: nn.ParameterDict
    curr_data: list[None | torch.Tensor]

    def __init__(
        self,
        nodes: list[nodes_.Node],
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

        self.stored_modules = nn.ModuleDict()
        self.stored_parameters = nn.ParameterDict()
        self.input_nodes = []
        self.output_nodes = []
        for node_id, node in enumerate(self.nodes):
            if isinstance(node, nodes_.OperatorNode):
                self.stored_modules[str(node_id)] = node.get_op()

            if isinstance(node, nodes_.ParameterNode):
                self.stored_parameters[str(node_id)] = nn.Parameter(torch.empty(1))

            if isinstance(node, nodes_.InputNode):
                self.input_nodes.append(node_id)

            if isinstance(node, nodes_.OutputNode):
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

        for str_node_id, param in self.stored_parameters.items():
            self.curr_data[int(str_node_id)] = param

        for node_id, node in enumerate(self.nodes):
            if isinstance(node, nodes_.ResponseInputNode | nodes_.LossOutputNode):
                break

            self.infer_node(node_id)

        output: list[torch.Tensor] = []
        for node_id in self.output_nodes:
            data = self.curr_data[node_id]
            assert data is not None, f"Output node {node_id} has not been inferred."

            output.append(data)
        return output

    def infer_node(self, node_id: int) -> None:
        """Infer the value of a node.

        Args:
            node_idx: Index of the node.
            data: List of tensors for each node.
        """
        node = self.nodes[node_id]

        if self.curr_data[node_id] is not None:
            return

        if isinstance(node, nodes_.OutputNode):
            input_node_ids = self.rev_adjacency_list[node_id]

            assert len(input_node_ids) == 1, "Output nodes should only have one incoming edge."
            input_node_id = input_node_ids[0]
            self.curr_data[node_id] = self.curr_data[input_node_id]
            return

        if isinstance(node, nodes_.OperatorNode):
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
        if isinstance(node, nodes_.DataNode):
            dot.node(str(node_idx), label=label)
        elif isinstance(node, nodes_.OperatorNode):
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

