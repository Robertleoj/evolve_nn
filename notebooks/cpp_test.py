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
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import project.foundation.graph as cpp_graph_
import project.graph.compiled as compiled_
import project.graph.graph as graph_
from IPython.display import display
from project.evolution.initialize import initialize_population
from project.evolution.select_and_mutate import select_and_mutate
from project.foundation.train import train_mse_single_pass, train_population
from project.type_defs import EvolutionConfig
from project.utils.paths import get_results_dir

# %%
evolution_config = EvolutionConfig(
    mutate_num_mutations=False,
    max_num_mutations=5,
    population_size=100,
    top_k_stay=3,
    num_epochs_training=1000,
    num_edges_weight=1e-3,
    num_parameters_weight=1e-3,
    softmax_temp=0.2,
    max_num_subgraphs=0,
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

test_spec = {
    "node_specs": [
        {"name": "input"},
        {"name": "parameter"},
        {"name": "prod"},
        {"name": "output"},
    ],
    "rev_adj_list": [[], [], [0, 1], [2]],
    "input_node_order": [0],
    "output_node_order": [3],
}


# %%
# define the problem: Sine wave
x = np.linspace(0, 1, 100)
y_clean = np.sin(x * np.pi * 2)
# y_clean = torch.sin(x ** 2)
# y_clean =  (x - 0.2) * (x - 0.8) * (x - 1.4)
y = y_clean + 0.1 * np.random.randn(*x.shape)

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")

# %%
g = graph_.make_graph(**test_spec)

# %%
graph_.show_graph(g)

# %%
comp_cpp = compiled_.to_cpp_compiled(g)

# %%
out = comp_cpp.forward([x])
plt.plot(x, out[0])

# %%
out

# %%
train_mse_single_pass(graph=comp_cpp, input=[x], target=[y], num_epochs=1000, learning_rate=1e-3)

# %%
comp_cpp.forward([x])


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


def evaluate_population(population, evolution_config):
    compiled_population = [compiled_.to_cpp_compiled(individual.graph) for individual in population]
    learning_rates = [float(individual.training_hp.lr) for individual in population]

    train_start = default_timer()
    train_population(
        compiled_population,
        [x],
        [y],
        evolution_config.num_epochs_training,
        learning_rates,
        12,
    )
    train_end = default_timer()
    print("Time taken to train population: {}".format(train_end - train_start))

    out = []
    for individual, compiled in zip(population, compiled_population):
        out.append(evaluate_net(individual.graph, compiled, x, y, evolution_config))

    fitness_scores, y_hats = zip(*out)

    # replace nan values with a high value
    fitness_scores = replace_invalid_with_high(fitness_scores)

    return fitness_scores, y_hats


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
    svgs = [graph_.get_graph_svg(ind.graph) for ind in ind_to_show]

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

        select_and_mutate_start = default_timer()
        new_population, recombination_probs = select_and_mutate(population, fitness_scores, evolution_config)
        select_and_mutate_end = default_timer()
        print("Time taken to select and mutate: {}".format(select_and_mutate_end - select_and_mutate_start))

        report_start = default_timer()
        report_data(population, fitness_scores, y_hats, recombination_probs, path, i)
        report_end = default_timer()
        print("Time taken to report data: {}".format(report_end - report_start))

        population = new_population


# %%
population = initialize_population(init_spec, evolution_config)

# %%
evolved = evolve(population, 1000, evolution_config)
population = [individual for _, individual, _ in evolved]
