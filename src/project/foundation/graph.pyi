import numpy

class AddNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.AddNode) -> None"""

class CompiledGraph:
    def __init__(self, nodes_topsorted: list[Node], rev_adj_list: list[list[int]], input_order: list[int], output_order: list[int]) -> None:
        """__init__(self: foundation.graph.CompiledGraph, nodes_topsorted: list[foundation.graph.Node], rev_adj_list: list[list[int]], input_order: list[int], output_order: list[int]) -> None"""
    def forward(self, arg0: list[numpy.ndarray]) -> list[numpy.ndarray]:
        """forward(self: foundation.graph.CompiledGraph, arg0: list[numpy.ndarray]) -> list[numpy.ndarray]"""

class DataNode(Node):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class ExpNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.ExpNode) -> None"""

class GELUNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.GELUNode) -> None"""

class InputNode(DataNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.InputNode) -> None"""

class LogNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.LogNode) -> None"""

class LossOutputNode(DataNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.LossOutputNode) -> None"""

class NegNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.NegNode) -> None"""

class Node:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class OperatorNode(Node):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class OutputNode(DataNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.OutputNode) -> None"""

class ParameterNode(DataNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.ParameterNode) -> None"""

class ProdNode(OperatorNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.ProdNode) -> None"""

class ResponseInputNode(DataNode):
    def __init__(self) -> None:
        """__init__(self: foundation.graph.ResponseInputNode) -> None"""
