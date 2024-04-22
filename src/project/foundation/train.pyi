import foundation.graph
import numpy

def mse_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int) -> None:
    """mse_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int) -> None

    Train a population of models
    """
def mse_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float) -> None:
    """mse_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float) -> None

    Train a model using MSE loss
    """
def response_regression_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int) -> None:
    """response_regression_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int) -> None

    Train a population of models with evolved loss computation
    """
def response_regression_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float) -> None:
    """response_regression_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float) -> None

    Train a model with evolved loss computation
    """
