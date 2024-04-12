"""Useful types."""
from dataclasses import dataclass



@dataclass
class EvolutionConfig:
    mutate_num_mutations: bool = True
    max_num_mutations: int | None = None
    population_size: int = 1000
    top_k_stay: int = 10
    num_epochs_training: int = 400
    num_edges_weight: float = 0.01
    num_parameters_weight: float = 0.01

