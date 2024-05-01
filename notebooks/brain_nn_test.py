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
#     display_name: project-OqwmfHuU-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
from project.brain_nn.brain import BrainNode, make_update_weights, make_brain, show_brain
from project.brain_nn.evolution.mutations import mutate_brain, recombine_brains
from project.brain_nn.type_defs import EvolutionConfig

# %%
ev_config = EvolutionConfig(
    num_transmitters=2,
    softmax_temp=0.3,
    init_weight_update_rate=0.05,
    weight_update_depth=2
)

# %%
b = BrainNode(2, ev_config.num_transmitters)
display(b.transform_weights)
display(b.input_weights)

# %%
update_weights = make_update_weights(ev_config.num_transmitters, ev_config.weight_update_depth)

# %%
inputs = [
    np.random.rand(ev_config.num_transmitters),
    np.random.rand(ev_config.num_transmitters)
]
print(inputs)

# %%
b.forward(inputs, update_weights=update_weights, weight_update_rate=ev_config.init_weight_update_rate)

# %%
brain = make_brain(
    num_inputs=2,
    num_outputs=1,
    num_response_input_nodes=1,
    num_transmitters=ev_config.num_transmitters,
    update_rate=ev_config.init_weight_update_rate,
)


# %%
brain.reverse_adj_list

# %%
brain.step(
    [
        np.array([0, 0]),
        np.array([1, 2])
    ], 
    [
        np.array([3, 4]), 
        np.array([5, 6])
    ])

# %%
show_brain(brain)

# %%
mutated = brain

# %%
mutated = mutate_brain(mutated)
show_brain(mutated)

# %%
mutated.step(
    [
        np.array([0, 0]),
        np.array([1, 2])
    ], 
    [
        np.array([3, 4]), 
        np.array([5, 6])
    ])

# %%
mut2 = mutate_brain(mutated)
show_brain(mut2)

# %%
rec = recombine_brains(mutated, mut2)
show_brain(rec)

# %%
