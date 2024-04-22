import foundation.graph
import numpy

def one_d_regression_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int, evolving_loss: bool, early_stop_k: int) -> None:
    """one_d_regression_train_population(population: list[foundation.graph.CompiledGraph], input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: list[float], num_threads: int, evolving_loss: bool, early_stop_k: int) -> None

    Train a population of models
    """
def one_d_regression_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float, evolving_loss: bool, early_stop_k: int) -> None:
    """one_d_regression_train_single_pass(graph: foundation.graph.CompiledGraph, input: list[numpy.ndarray], target: list[numpy.ndarray], num_epochs: int, learning_rate: float, evolving_loss: bool, early_stop_k: int) -> None

    Train a model using MSE loss
    """
