import foundation.graph
import numpy

def train_mse_single_pass(
    graph: foundation.graph.CompiledGraph,
    input: list[numpy.ndarray],
    target: list[numpy.ndarray],
    num_epochs: int,
    learning_rate: float,
) -> None:
    """train_mse_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float) -> None

    Train a model using MSE loss
    """

def train_population(
    population: list[foundation.graph.CompiledGraph],
    input: list[numpy.ndarray],
    target: list[numpy.ndarray],
    num_epochs: int,
    learning_rate: list[float],
    num_threads: int,
) -> None:
    """train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int) -> None

    Train a population of models
    """
