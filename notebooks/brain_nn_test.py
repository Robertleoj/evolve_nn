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
from project.brain_nn.brain import BrainNode, make_update_weights, make_brain

# %%
NUM_TRANSMITTERS = 4
WEIGHT_UPDATE_RATE = 0.01
WEIGHT_UPDATE_DEPTH = 2

# %%
b = BrainNode(2, NUM_TRANSMITTERS)
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
brain = make_brain(
    num_inputs=2,
    num_outputs=1,
    num_response_input_nodes=1,
    num_transmitters=NUM_TRANSMITTERS,
)


# %%
brain
