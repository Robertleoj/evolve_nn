# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: project-xOhHZUaJ-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import torch
import torch.nn as nn
from einops import rearrange
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from project.graph_novec import graph, nodes, compiled, mutation
from project.graph_novec.graph import Graph, show_graph
import math

# %%
POPULATION_SIZE = 500
NUM_EPOCHS = 1000
NUM_MUTATIONS = 10


# %%
def mutate(graph: graph.Graph, num_mutations=NUM_MUTATIONS) -> graph.Graph:
    mutated = graph

    mutation_functions = [
        mutation.expand_edge,
        mutation.add_edge,
        mutation.add_parameter,
        mutation.delete_operator,
        mutation.delete_edge,
        mutation.delete_parameter
    ]

    mutation_probabilities = [
        0.3,
        0.6,
        0.3,
        0.1,
        0.25,
        0.1
    ]

    mutations_performed = 0
    while mutations_performed < num_mutations:
        mutation_function = random.choices(mutation_functions, weights=mutation_probabilities)[0]
        mutated, changed = mutation_function(mutated)

        try:
            mutation.check_validity(mutated)
        except Exception as e:
            print("Tried to mutate with {} but failed".format(mutation_function))
            print(e)
            show_graph(mutated)
            raise e

        if changed:
            mutations_performed += 1

    return mutated


# %%
def initialize_population(population_size):
    init_spec = {
        "node_names": [
            'input',
            'output',           
        ],
        "edge_list": [
            (0, 1)
        ]
    }

    population = [mutate(graph.make_graph(**init_spec)) for _ in range(population_size)]
    return population


# %%
# define the problem: Sine wave

x = torch.linspace(0, 1, 100)
y_clean = torch.sin(x * torch.pi * 2 )
y = y_clean + 0.1 * torch.randn(x.size())

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")


# %%
def train_single_net(net: compiled.CompiledGraph) -> None:
    if len(list(net.parameters())) == 0:
        return

    try:
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        x_in = rearrange(x, "b -> b")
        targ = rearrange(y, "b -> b")

        for i in range(NUM_EPOCHS):
            optimizer.zero_grad()
            output = net([x_in])[0]
            if output.shape != targ.shape:
                raise ValueError("Shapes don't match")
            loss = loss_fn(output, targ)
            loss.backward()
            optimizer.step()
    except Exception as e:
        print("Failed to train net")
        print(e)
        compiled.show_compiled(net)

def replace_invalid_with_high(values, high_value=100):
    return [high_value if math.isinf(x) or math.isnan(x) else x for x in values]

def evaluate_population(population, num_edges_weight=0.01, num_parameters_weight=0.01):
    num_edges = [len(net.edge_list) for net in population]

    compiled_nets = [compiled.CompiledGraph.from_graph(net) for net in population]
    num_parameters = [len(list(net.parameters())) for net in compiled_nets]

    for compiled_net in tqdm(compiled_nets):
        train_single_net(compiled_net)

    # now all individuals are trained

    # evaluate each individual
    fitness_scores = []
    for i, compiled_net in enumerate(compiled_nets):
        compiled_net.eval()
        with torch.no_grad():
            output = compiled_net([x])
            loss = torch.nn.MSELoss()(output[0], y)

        fitness_scores.append(loss.item() + num_edges_weight * num_edges[i] + num_parameters_weight * num_parameters[i])

    # replace nan values with a high value
    fitness_scores = replace_invalid_with_high(fitness_scores)

    return fitness_scores, compiled_nets

def select_and_mutate(population, fitness_scores):
    # Zip together the population and fitness scores, then sort them by loss
    paired_pop = list(zip(population, fitness_scores))
    paired_pop.sort(key=lambda x: x[1], reverse=False)  # Sort by fitness score, low to high

    # Keep the best individual unchanged
    best_individual = paired_pop[0][0]

    # Select the top half, excluding the best one since it's already included
    top_half = [individual for individual, score in paired_pop[:(len(paired_pop) // 2)]]

    # Clone and mutate to fill up the next generation, starting with the best individual
    next_generation = [best_individual]  # Start with the best individual unchanged
    for individual in top_half:
        next_generation.append(individual)  # Clone gets mutated
        if len(next_generation) < len(population) - 1:  # Check to not exceed the original size
            next_generation.append(mutate(individual))  # Second clone also gets mutated

    # If there's still room (for odd-sized populations), mutate another clone of the second-best
    if len(next_generation) < len(population):
        next_generation.append(mutate(top_half[0]))

    return next_generation



# %%
def evolve(population, iterations = 10, num_edges_weight=0.01, num_parameters_weight=0.01):
    for i in range(iterations):
        fitness_scores, _ = evaluate_population(population, num_edges_weight, num_parameters_weight)
        best_score = min(fitness_scores)
        print(f"Generation {i}, best score: {best_score}")
        best_individual = population[fitness_scores.index(best_score)]
        graph.show_graph(best_individual)
        population = select_and_mutate(population, fitness_scores)
    
    fitness_scores, compiled = evaluate_population(population, num_edges_weight, num_parameters_weight)
    scores_and_individuals = zip(fitness_scores, population, compiled)
    scores_and_individuals = sorted(scores_and_individuals, key=lambda x: x[0], reverse=False)
    return scores_and_individuals


# %%
population = initialize_population(POPULATION_SIZE)

# %%
evolved = evolve(population, 1000, num_edges_weight=1e-6, num_parameters_weight=1e-6)
population = [individual for _, individual, _ in evolved]

# %%
best_score, best_evolved, best_compiled = evolved[0]
best_score

# %%
graph.show_graph(best_evolved)

# %%
with torch.no_grad():
    y_hat = best_compiled([x])[0]

plt.plot(x, y_hat, color="green")
plt.scatter(x, y)

