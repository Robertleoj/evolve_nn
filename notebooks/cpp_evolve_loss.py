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

import numpy as np
from project.evolution.initialize import initialize_population
from project.type_defs import EvolutionConfig
from project.tasks.one_d_regression import get_init_spec, evolve, generate_target_poly

# %%
evolution_config = EvolutionConfig(
    mutate_num_mutations=False,
    max_num_mutations=5,
    population_size=5000,
    top_k_stay=3,
    num_epochs_training=500,
    num_edges_weight=2e-5,
    num_parameters_weight=2e-5,
    softmax_temp=1.0,
    max_num_subgraphs=0,
)

# %%
evolve_loss = True

# %%
init_spec = get_init_spec(evolve_loss=evolve_loss)
print(init_spec)
population = initialize_population(init_spec, evolution_config)

# %%
x = np.linspace(0, 1, 100)
evolved = evolve(population, 1000, evolution_config, x, evolve_loss=evolve_loss, generate_target_fn=generate_target_poly, num_targets=10)
population = [individual for _, individual, _ in evolved]
