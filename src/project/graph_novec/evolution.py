from project.type_defs import EvolutionConfig
from project.graph_novec.individual import Individual, random_individual
import random

def initialize_population(init_spec: dict, evolution_config: EvolutionConfig) -> list[Individual]:


    population = [
        random_individual(
            init_spec, 
            evolution_config
        )
        for _ in range(evolution_config.population_size)
    ]
    return population