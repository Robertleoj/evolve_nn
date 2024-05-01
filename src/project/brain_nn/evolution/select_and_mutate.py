from project.brain_nn.brain import Brain
import numpy as np
import random
from itertools import cycle
from project.brain_nn.type_defs import EvolutionConfig
from project.brain_nn.evolution.mutations import mutate_brain, recombine_brains

def select_and_mutate(
    population: list[Brain], 
    fitness_scores: list[float], 
    evolution_config: EvolutionConfig, 
    minimize=False
) -> tuple[list[Brain], list[float]]:
    """Selection and mutation of individuals based on their fitness scores.

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
    # normalize the fitness scores between 0 and 1
    scores_np = scores_np - np.min(scores_np)
    scores_np = scores_np / np.max(scores_np)

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

            recombined = recombine_brains(individual, other)
            mutated_offspring = mutate_brain(recombined)
            next_generation.append(mutated_offspring)

    return next_generation, probabilities

