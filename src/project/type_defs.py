"""Useful types."""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class EvolutionConfig:
    mutate_num_mutations: bool = True
    max_num_mutations: int | None = None
    population_size: int = 1000
    top_k_stay: int = 10
    num_epochs_training: int = 400
    num_edges_weight: float = 0.01
    num_parameters_weight: float = 0.01
    softmax_temp: float = 1.0
    max_subgraph_depth: int = 1
    max_num_subgraphs: int = 5


@dataclass
class GraphMutHP:
    max_num_mutations: int
    mutation_probabilities: dict[str, float]
    subgraph_mutation_probabilities: dict[str, float]
    operator_probabilities: dict[str, float]
    subgraph_operator_probabilities: dict[str, float]
    max_subgraph_depth: int = 1
    max_num_subgraphs: int = 5


@dataclass
class TrainingHP:
    """Training hyperparameters."""

    lr: float
    momentum: float


class NumpyModule:
    """A module that takes a list of numpy arrays and returns a numpy array."""

    def __call__(self, args: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError