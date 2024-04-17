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
import math
import random
from datetime import datetime
from itertools import cycle, repeat
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from einops import rearrange
from project.evolution.initialize import initialize_population
from project.graph.graph import CompiledGraph, get_graph_svg, show_compiled, show_graph
from project.type_defs import EvolutionConfig
from project.utils.paths import get_results_dir
from project.variation_ops import mutate_individual, recombine_individuals
from tqdm import tqdm

# %%
evolution_config = EvolutionConfig(
    mutate_num_mutations=False,
    max_num_mutations=5,
    population_size=1000,
    top_k_stay=3,
    num_epochs_training=1000,
    num_edges_weight=1e-3,
    num_parameters_weight=1e-3,
    softmax_temp=0.2,
)

# %%
init_spec = {
    "node_specs": [
        {"name": "input"},
        {"name": "output"},
    ],
    "rev_adj_list": [[], [0]],
    "input_node_order": [0],
    "output_node_order": [1],
}


# %%
# define the problem: Sine wave
x = torch.linspace(0, 1, 100)
y_clean = torch.sin(x * torch.pi * 2)
# y_clean = torch.sin(x ** 2)
# y_clean =  (x - 0.2) * (x - 0.8) * (x - 1.4)
y = y_clean + 0.05 * torch.randn(x.size())

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")


# %%
def evaluate_net(graph_net, compiled_net, x, y, evolution_config):
    compiled_net.eval()
    with torch.no_grad():
        output = compiled_net([x])
        loss = torch.nn.MSELoss()(output[0], y)

    num_edges = len(graph_net.edge_list)
    num_parameters = len(graph_net.parameter_nodes())

    edge_weight = evolution_config.num_edges_weight
    num_parameters_weight = evolution_config.num_parameters_weight

    return (loss.cpu().item() + edge_weight * num_edges + num_parameters_weight * num_parameters), [
        o.cpu().numpy() for o in output
    ]


# %%


def train_eval_single_net(args) -> float:
    torch.set_num_threads(1)
    individual, x, y, evolution_config = args
    graph = individual.graph

    compiled = CompiledGraph.from_graph(graph)

    x = torch.tensor(x)
    y = torch.tensor(y)

    learning_rate = individual.training_hp.lr

    # net = compiled.compile(graph_net)
    if len(list(compiled.parameters())) > 0:
        try:
            loss_fn = torch.nn.MSELoss()
            # optimizer = torch.optim.SGD(compiled.parameters(), lr=learning_rate, momentum=momentum)
            optimizer = torch.optim.Adam(compiled.parameters(), lr=learning_rate)
            x_in = rearrange(x, "b -> b")
            targ = rearrange(y, "b -> b")

            default_timer()
            for i in range(evolution_config.num_epochs_training):
                optimizer.zero_grad()
                output = compiled([x_in])[0]
                if output.shape != targ.shape:
                    raise ValueError("Shapes don't match")
                loss = loss_fn(output, targ)
                loss.backward()
                optimizer.step()

            default_timer()
            # print("Time taken to train net: {}".format(end_time - start_time))

        except Exception as e:
            print("Failed to train net")
            print(e)
            show_compiled(compiled)
            raise e

    default_timer()
    eval_out = evaluate_net(graph, compiled, x, y, evolution_config)
    default_timer()
    # print("Time taken to evaluate net: {}".format(end_time - start_time))
    return eval_out


# %%


def replace_invalid_with_high(values, high_value=100):
    return [high_value if math.isinf(x) or math.isnan(x) else x for x in values]


def evaluate_population(population, evolution_config):
    x_np = x.numpy()
    y_np = y.numpy()

    args = zip(population, repeat(x_np), repeat(y_np), repeat(evolution_config))

    # Make sure args is iterable of iterables (e.g., list of tuples)
    with mp.Pool(16) as p:
        out = []
        for result in tqdm(p.imap(train_eval_single_net, args), desc="Evaluating population", total=len(population)):
            out.append(result)

    fitness_scores, y_hats = zip(*out)

    # replace nan values with a high value
    fitness_scores = replace_invalid_with_high(fitness_scores)

    return fitness_scores, y_hats


# %%

# def select_and_mutate(population, fitness_scores, evolution_config):
#     # Zip together the population and fitness scores, then sort them by loss
#     paired_pop = list(zip(population, fitness_scores))
#     paired_pop.sort(key=lambda x: x[1], reverse=False)  # Sort by fitness score, low to high

#     # Select the top half, excluding the best one since it's already included
#     top_half = [individual for individual, score in paired_pop[:(len(paired_pop) // 2)]]

#     # Clone and mutate to fill up the next generation, starting with the best individual
#     # next_generation = [best_individual]  # Start with the best individual unchanged
#     next_generation = top_half[:evolution_config.top_k_stay] # Start with the best individual unchanged

#     for individual in cycle(top_half):
#         too_many = False
#         for _ in range(2):
#             if len(next_generation) >= len(population):  # Check to not exceed the original size
#                 too_many = True
#                 break
#             other = random.choice(top_half)
#             recombined = recombine_individuals(individual, other, evolution_config)
#             rec_mut = mutate_individual(recombined, evolution_config)
#             next_generation.append(rec_mut)
#         if too_many:
#             break

#     return next_generation


def select_and_mutate(population, fitness_scores, evolution_config):
    # Apply softmax to negative fitness scores directly to favor individuals with lower scores
    # normalize the fitness scores between 0 and 1

    softmax_temp = evolution_config.softmax_temp
    exp_scores = np.exp(-np.array(fitness_scores) / softmax_temp)  # Exponentiate the negative fitness scores
    probabilities = exp_scores / np.sum(exp_scores)  # Softmax probabilities

    # Use softmax probabilities to perform weighted selection
    selected_indices = random.choices(range(len(population)), weights=probabilities, k=len(population) // 2)
    selected_population = [population[i] for i in selected_indices]

    sorted_by_score = sorted(list(zip(population, fitness_scores)), key=lambda x: x[1])
    next_generation = [individual for individual, score in sorted_by_score[: evolution_config.top_k_stay]]

    # Clone and mutate to fill up the next generation
    for individual in cycle(selected_population):
        if len(next_generation) >= len(population):
            break
        for _ in range(2):  # Assume each selected individual can produce two offspring
            if len(next_generation) >= len(population):
                break
            other = random.choice(selected_population)
            recombined = recombine_individuals(individual, other, evolution_config)
            mutated_offspring = mutate_individual(recombined, evolution_config)
            next_generation.append(mutated_offspring)

    return next_generation, probabilities


# %%


def report_data(population, fitness_scores, y_hats, recombination_probs, folder_path: Path, generation: int) -> None:
    generation_path = folder_path / f"generation_{generation}"
    generation_path.mkdir(parents=True, exist_ok=True)

    best_score = min(fitness_scores)
    print(f"Generation {generation}, best score: {best_score}")

    best_score_idx = fitness_scores.index(best_score)
    y_hats[best_score_idx]

    best_individual = population[best_score_idx]
    print(best_individual.training_hp)
    print(best_individual.graph_mut_hps)

    ordered = sorted(zip(fitness_scores, population, y_hats), key=lambda x: x[0])
    num_to_show = 10
    ordered_to_show = ordered[:num_to_show]

    _, ind_to_show, y_hats_to_show = zip(*ordered_to_show)
    svgs = [get_graph_svg(ind.graph) for ind in ind_to_show]

    graphs_path = generation_path / "graphs"
    graphs_path.mkdir()
    for i, (svg, y_hat) in enumerate(zip(svgs, y_hats_to_show)):
        (graphs_path / f"individual_{i}_graph.svg").write_text(svg.data)
        fig, ax = plt.subplots()
        ax.plot(x, y_hat[0], color="green")
        ax.scatter(x, y)
        plt.savefig(graphs_path / f"individual_{i}_plot.png")
        if i == 0:
            (generation_path / "best_individual_plot.svg").write_text(svg.data)
            display(svg)
            plt.savefig(generation_path / "best_individual_plot.png")
            plt.show()
        plt.close(fig)

    probabilities: list[float] = recombination_probs.tolist()
    probabilities.sort(reverse=True)

    fig, ax = plt.subplots()
    ax.bar(range(len(probabilities)), probabilities)
    plt.savefig(generation_path / "probabilities.png")
    plt.show()
    plt.close(fig)


def generate_folder_name() -> str:
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y%m%d_%H%M%S")
    return folder_name


def evolve(population, iterations, evolution_config, path: Path | None = None):
    if path is None:
        path = get_results_dir() / generate_folder_name()
        print(path)

    for i in range(iterations):
        fitness_scores, y_hats = evaluate_population(population, evolution_config)
        assert len(fitness_scores) == len(population)
        assert len(y_hats) == len(population)
        new_population, recombination_probs = select_and_mutate(population, fitness_scores, evolution_config)

        report_data(population, fitness_scores, y_hats, recombination_probs, path, i)

        population = new_population

    # fitness_scores, compiled = evaluate_population(population, num_edges_weight, num_parameters_weight)
    # scores_and_individuals = zip(fitness_scores, population, compiled)
    # scores_and_individuals = sorted(scores_and_individuals, key=lambda x: x[0], reverse=False)
    # return scores_and_individuals


# %%
population = initialize_population(init_spec, evolution_config)

# %%
# ind = population[3]
# g = ind.graph
for ind in population[:3]:
    show_graph(ind.graph)

# %%
evolved = evolve(population, 1000, evolution_config)
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
