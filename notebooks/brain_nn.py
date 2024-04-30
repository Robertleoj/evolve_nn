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
from brain_nn.brain import BrainNode, make_update_weights

# %%
NUM_TRANSMITTERS = 4
WEIGHT_UPDATE_RATE = 0.05
WEIGHT_UPDATE_DEPTH = 2

# %%
b = BrainNode(2, NUM_TRANSMITTERS)

# %%
display(b.transform_weights)
display(b.input_weights)

# %%
update_weights = make_update_weights(NUM_TRANSMITTERS, WEIGHT_UPDATE_DEPTH)

# %%
inputs = [
    np.random.rand(NUM_TRANSMITTERS),
    np.random.rand(NUM_TRANSMITTERS)
]
print(inputs)

# %%
b.forward(inputs, update_weights=update_weights, weight_update_rate=WEIGHT_UPDATE_RATE)


# %%
class Brain:
    id_to_node: dict[str, BrainNode]
    reverse_adj_list: dict[str, set[str]]

    input_nodes: list[str]
    response_input_nodes: list[str]
    output_nodes: list[str]

    curr_outputs: dict[str, np.ndarray]

    output_weights: np.ndarray


# %%
