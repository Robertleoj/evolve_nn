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
from datetime import datetime
import random
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
from itertools import count
import numpy as np
import project.foundation.graph as cpp_graph_
import project.graph.compiled as compiled_
import project.graph.graph as graph_
from IPython.display import display
from project.evolution.initialize import initialize_population
from project.evolution.select_and_mutate import select_and_mutate
import project.foundation.train as cpp_train_
from project.type_defs import EvolutionConfig
from project.utils.paths import get_results_dir

# %%
evolution_config = EvolutionConfig(
    mutate_num_mutations=False,
    max_num_mutations=2,
    population_size=200,
    top_k_stay=20,
    num_epochs_training=500,
    num_edges_weight=2e-4,
    num_parameters_weight=2e-4,
    # num_edges_weight=2e-3,
    # num_parameters_weight=2e-3,
    # num_edges_weight=0,
    # num_parameters_weight=0,
    softmax_temp=0.2,
    max_num_subgraphs=0,
)

# %%
init_spec = {
    "node_specs": [
        {"name": "input"},
        {'name': 'parameter'},
        {'name': "add"},
        {"name": "output"},
        {"name": "response_input"},
        {"name": "add"},
        {"name": "loss_output"},
    ],
    "rev_adj_list": [
        [],  # 0 input
        [],  # 1 parameter
        [0, 1],  # 2 add
        [2],   # 3 output
        [],  # 4 response_input,
        [3, 4],  # 5 add
        [5],  # 6 loss_output
    ],
    "input_node_order": [0],
    "output_node_order": [3],
    "response_input_node_order": [4],
}




# %%
population = initialize_population(init_spec, evolution_config)


# %%
def generate_target_poly(x: np.ndarray) -> np.ndarray:
    order = random.randint(1, 5)

    coeff = np.random.randn(order + 1)

    y = np.zeros_like(x)
    for i in range(order + 1):
        y += coeff[i] * x ** i

    return y

def generate_target_sin(x: np.ndarray) -> np.ndarray:
    freq = random.uniform(1.0, 2.3)
    phase = random.uniform(0, 2 * math.pi)
    y = np.sin(x * freq * math.pi * 2 + phase)
    return y

def generate_target(x: np.ndarray) -> np.ndarray:
    return generate_target_poly(x)

    functions = [
        generate_target_poly,
        generate_target_sin,
    ]

    return random.choice(functions)(x) + 0.01 * np.random.randn(*x.shape)

    

# %%
def evaluate_net(
    graph_net: graph_.Graph,
    compiled_net: cpp_graph_.CompiledGraph,
    x: np.ndarray,
    y: np.ndarray,
    evolution_config: EvolutionConfig,
):
    output = compiled_net.forward([x])
    loss = np.mean((output[0] - y) ** 2)

    num_edges = len(graph_net.edge_list)
    num_parameters = len(graph_net.parameter_nodes())

    edge_weight = evolution_config.num_edges_weight
    num_parameters_weight = evolution_config.num_parameters_weight

    return (loss + edge_weight * num_edges + num_parameters_weight * num_parameters), output


# %%


def replace_invalid_with_high(values, high_value=100):
    return [high_value if math.isinf(x) or math.isnan(x) else x for x in values]


def evaluate_population(population, x, targets, evolution_config):
    fitness_scores_all = []
    y_hat_all = []

    for targ in targets:
        compile_start = default_timer()
        compiled_population = [compiled_.to_cpp_compiled(individual.graph) for individual in population]
        compile_end = default_timer()
        print("Time taken to compile population: {}".format(compile_end - compile_start))
        learning_rates = [float(individual.training_hp.lr) for individual in population]


        train_start = default_timer()
        cpp_train_.response_regression_train_population(
            compiled_population,
            [x],
            [targ],
            evolution_config.num_epochs_training,
            learning_rates,
            16,
        )
        train_end = default_timer()
        print("Time taken to train population: {}".format(train_end - train_start))

        evaluate_start = default_timer()
        out = []
        for individual, compiled in zip(population, compiled_population):
            out.append(evaluate_net(individual.graph, compiled, x, targ, evolution_config))
        evaluate_end = default_timer()
        print("Time taken to evaluate population: {}".format(evaluate_end - evaluate_start))

        fitness_scores, y_hats = zip(*out)

        # replace nan values with a high value
        fitness_scores = replace_invalid_with_high(fitness_scores)
        fitness_scores_all.append(fitness_scores)
        y_hat_all.append(y_hats)

    fitness_scores = np.mean(fitness_scores_all, axis=0)

    return fitness_scores, y_hat_all


# %%


def report_data(population, fitness_scores, target, y_hats, recombination_probs, folder_path: Path, generation: int) -> None:
    generation_path = folder_path / f"generation_{generation}"
    generation_path.mkdir(parents=True, exist_ok=True)



    best = min(list(enumerate(fitness_scores)), key=lambda x: x[1])

    best_index = best[0]
    best_score = best[1]

    print(f"Generation {generation}, best score: {best_score}")

    best_individual = population[best_index]
    print(best_individual.training_hp)
    print(best_individual.graph_mut_hps)


    svg = graph_.get_graph_svg(best_individual.graph)
    (generation_path / "best_individual_plot.svg").write_text(svg.data)
    display(svg)

    probabilities: list[float] = recombination_probs.tolist()
    probabilities.sort(reverse=True)

    fig, ax = plt.subplots()
    ax.bar(range(len(probabilities)), probabilities)
    plt.savefig(generation_path / "probabilities.png")
    plt.show()
    plt.close(fig)

    scale = 5
    fig, axes = plt.subplots(1, len(target), figsize=(len(target) * 5, 5))
    for (i, targ), ax in zip(enumerate(target), axes.flatten()):
        y_hat = y_hats[i][best_index][0]
        ax.scatter(x, targ)
        ax.plot(x, y_hat, color="red")

    fig.tight_layout()

    plt.savefig(generation_path / "best_individual_predictions.png")
    plt.show()
    plt.close(fig)


def generate_folder_name() -> str:
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y%m%d_%H%M%S")
    return folder_name


# %%


def evolve(population, iterations, evolution_config, x, path: Path | None = None):
    
    if path is None:
        path = get_results_dir() / generate_folder_name()
        print(path)

    for i in range(iterations):

        targets = [
            generate_target(x)
            for _ in range(5)
        ]

        fitness_scores, y_hats = evaluate_population(population, x, targets, evolution_config)
        assert len(fitness_scores) == len(population)

        select_and_mutate_start = default_timer()
        new_population, recombination_probs = select_and_mutate(population, fitness_scores, evolution_config)
        select_and_mutate_end = default_timer()
        print("Time taken to select and mutate: {}".format(select_and_mutate_end - select_and_mutate_start))

        if i % 1 == 0:
            report_start = default_timer()
            report_data(population, fitness_scores, targets, y_hats, recombination_probs, path, i)
            report_end = default_timer()
            print("Time taken to report data: {}".format(report_end - report_start))

        population = new_population


# %%
x = np.linspace(0, 1, 100)
evolved = evolve(population, 1000, evolution_config, x)
population = [individual for _, individual, _ in evolved]
