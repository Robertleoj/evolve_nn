from dataclasses import dataclass


@dataclass
class EvolutionConfig:
    num_episodes_per_training: int = 50
    steps_per_action: int = 5
    population_size: int = 100
    num_transmitters: int = 3
    softmax_temp: float = 0.3
    init_weight_update_rate: float = 0.05
    weight_update_depth: int = 2
    top_k_stay: int = 5