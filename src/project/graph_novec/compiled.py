import torch.nn as nn
import torch
from project.graph_novec.nodes import DataNode, InputNode, Node, OutputNode, ParameterNode
from project.graph_novec.graph import topsort_edge_list, reverse_adjacency_list, Graph


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
