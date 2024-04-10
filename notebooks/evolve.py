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
import torch.multiprocessing as mp
from einops import rearrange
import random
from tqdm import tqdm
from timeit import default_timer
from itertools import repeat, cycle
import matplotlib.pyplot as plt
from project.graph_novec.individual import Individual, mutate_individual, recombine_individuals, random_individual
from project.graph_novec.graph import Graph, show_graph
from project.graph_novec.compiled import CompiledGraph, show_compiled
import math

# %%
POPULATION_SIZE = 1000
NUM_EPOCHS = 700
TOP_K = 10


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

    population = [random_individual(init_spec) for _ in range(population_size)]
    return population


# %%
# define the problem: Sine wave

x = torch.linspace(0, 1, 100)
y_clean = torch.sin(x * torch.pi * 2 )
# y_clean = torch.sin(x ** 2)
# y_clean =  (x - 0.2) * (x - 0.8) * (x - 1.4)
y = y_clean + 0.05 * torch.randn(x.size())

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")


# %%
def evaluate_net(
    graph_net,
    compiled_net, 
    x, 
    y, 
    obj_hp
):

    compiled_net.eval()
    with torch.no_grad():
        output = compiled_net([x])
        loss = torch.nn.MSELoss()(output[0], y)

    num_edges = len(graph_net.edge_list)
    num_parameters = len(graph_net.parameter_nodes())

    return (
        loss.cpu().item() 
        + obj_hp['num_edges_weight'] * num_edges 
        + obj_hp['num_parameters_weight'] * num_parameters
    ), [o.cpu().numpy() for o in output]


# %%

def train_eval_single_net(args) -> float:
    torch.set_num_threads(1)
    individual, x, y, obj_hp = args
    graph = individual.graph

    compiled = CompiledGraph.from_graph(graph)

    x = torch.tensor(x)
    y = torch.tensor(y)

    learning_rate = individual.training_hp.lr
    momentum = individual.training_hp.momentum

    # net = compiled.compile(graph_net)
    if len(list(compiled.parameters())) == 0:
        return float("inf"), []

    try:
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(compiled.parameters(), lr=learning_rate, momentum=momentum)
        x_in = rearrange(x, "b -> b")
        targ = rearrange(y, "b -> b")

        start_time = default_timer()
        for i in range(NUM_EPOCHS):
            optimizer.zero_grad()
            output = compiled([x_in])[0]
            if output.shape != targ.shape:
                raise ValueError("Shapes don't match")
            loss = loss_fn(output, targ)
            loss.backward()
            optimizer.step()

        end_time = default_timer()
        # print("Time taken to train net: {}".format(end_time - start_time))

        start_time = default_timer()
        eval_out = evaluate_net(graph, compiled, x, y, obj_hp)
        end_time = default_timer()
        # print("Time taken to evaluate net: {}".format(end_time - start_time))
        return eval_out
        
    except Exception as e:
        print("Failed to train net")
        print(e)
        show_compiled(compiled)



# %%

def replace_invalid_with_high(values, high_value=100):
    return [high_value if math.isinf(x) or math.isnan(x) else x for x in values]

def evaluate_population(population, num_edges_weight=0.01, num_parameters_weight=0.01):

    x_np = x.numpy()
    y_np = y.numpy()

    args = list(zip(
        population,
        repeat(x_np),
        repeat(y_np),
        repeat({
            'num_edges_weight': num_edges_weight,
            'num_parameters_weight': num_parameters_weight
        })
    ))

    # Make sure args is iterable of iterables (e.g., list of tuples)
    with mp.Pool(16) as p:
        out = []
        for result in tqdm(p.imap(train_eval_single_net, args), desc="Evaluating population", total=len(args)):
            out.append(result)

    fitness_scores, y_hats = zip(*out)

    # replace nan values with a high value
    fitness_scores = replace_invalid_with_high(fitness_scores)

    return fitness_scores, y_hats



# %%

def select_and_mutate(population, fitness_scores):
    # Zip together the population and fitness scores, then sort them by loss
    paired_pop = list(zip(population, fitness_scores))
    paired_pop.sort(key=lambda x: x[1], reverse=False)  # Sort by fitness score, low to high

    # Select the top half, excluding the best one since it's already included
    top_half = [individual for individual, score in paired_pop[:(len(paired_pop) // 2)]]

    # Clone and mutate to fill up the next generation, starting with the best individual
    # next_generation = [best_individual]  # Start with the best individual unchanged
    next_generation = top_half[:TOP_K] # Start with the best individual unchanged

    for individual in cycle(top_half):
        too_many = False
        for _ in range(2):
            if len(next_generation) >= len(population) - 1:  # Check to not exceed the original size
                too_many = True
                break
            other = random.choice(top_half)
            recombined = recombine_individuals(individual, other)
            rec_mut = mutate_individual(recombined)
            next_generation.append(rec_mut) 
        if too_many:
            break

    return next_generation



# %%
def evolve(population, iterations = 10, num_edges_weight=0.01, num_parameters_weight=0.01):
    for i in range(iterations):
        fitness_scores, y_hats = evaluate_population(population, num_edges_weight, num_parameters_weight)

        best_score = min(fitness_scores)
        print(f"Generation {i}, best score: {best_score}")

        best_score_idx = fitness_scores.index(best_score)
        best_y_hat = y_hats[best_score_idx]

        best_individual = population[best_score_idx]
        print(best_individual.training_hp)
        print(best_individual.graph_mut_hps)
        show_graph(best_individual.graph)
        
        plt.plot(x, best_y_hat[0], color="green")
        plt.scatter(x, y)
        plt.show()
        plt.close()
        population = select_and_mutate(population, fitness_scores)
    
    # fitness_scores, compiled = evaluate_population(population, num_edges_weight, num_parameters_weight)
    # scores_and_individuals = zip(fitness_scores, population, compiled)
    # scores_and_individuals = sorted(scores_and_individuals, key=lambda x: x[0], reverse=False)
    # return scores_and_individuals


# %%
population = initialize_population(POPULATION_SIZE)

# %%
evolved = evolve(population, 1000, num_edges_weight=1e-5, num_parameters_weight=1e-5)
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

