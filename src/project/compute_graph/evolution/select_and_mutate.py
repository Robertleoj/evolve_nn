import random
from itertools import cycle

import numpy as np
from compute_graph.evolution.individual import Individual
from compute_graph.type_defs import EvolutionConfig
from compute_graph.variation_ops import mutate_individual, recombine_individuals


def select_and_mutate(
    population: list[Individual], fitness_scores: list[float], evolution_config: EvolutionConfig, minimize=True
) -> tuple[list[Individual], list[float]]:
    """Select and mutate individuals based on their fitness scores.

    Args:
        population: List of individuals.
        fitness_scores: List of fitness scores.
        evolution_config: Evolution configuration.
        minimize: Whether to minimize the fitness scores.

    Returns:
        new_generation: List of individuals for the next generation.
        probabilities: Selection probabilities.
    """
    # Apply softmax to negative fitness scores directly to favor individuals with lower scores
    # normalize the fitness scores between 0 and 1
    softmax_temp = evolution_config.softmax_temp

    if minimize:
        fitness_scores = [-score for score in fitness_scores]

    scores_np = np.array(fitness_scores)

    exp_scores = np.exp(scores_np / softmax_temp)  # Exponentiate the negative fitness scores
    probabilities = exp_scores / np.sum(exp_scores)  # Softmax probabilities

    # Use softmax probabilities to perform weighted selection
    selected_indices = random.choices(range(len(population)), weights=probabilities, k=len(population) // 2)
    selected_population = [population[i] for i in selected_indices]
    selected_probabilities = [probabilities[i] for i in selected_indices]

    sorted_by_score = sorted(list(zip(population, fitness_scores)), key=lambda x: x[1])
    next_generation = [individual for individual, score in sorted_by_score[: evolution_config.top_k_stay]]

    # Clone and mutate to fill up the next generation
    for ind_idx, individual in cycle(enumerate(selected_population)):
        if len(next_generation) >= len(population):
            break
        for _ in range(2):  # Assume each selected individual can produce two offspring
            if len(next_generation) >= len(population):
                break

            # select other, not equal to individual
            selection_probabilities = selected_probabilities.copy()
            selection_probabilities[ind_idx] = 0
            other = random.choices(selected_population, weights=selection_probabilities)[0]
            recombined = recombine_individuals(individual, other, evolution_config)
            mutated_offspring = mutate_individual(recombined, evolution_config)
            next_generation.append(mutated_offspring)

    return next_generation, probabilities
