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
import compute_graph.graph.graph as graph
import compute_graph.graph.nodes as nodes
import torch

# %%
nodes.node_map

# %%
add_node = nodes.node_from_spec({"name": "add", "input_shapes": [(1, 2), (1, 2)], "shape": (1, 2)})

# %%
add_node([torch.tensor([1, 2]), torch.tensor([3, 4])])

# %%
# create a synthetic problem: add two numbers and apply ReLu
sample_graph_spec = {
    "node_specs": [
        {"name": "input", "shape": (1, 2)},
        {"name": "input", "shape": (1, 2)},
        {"name": "add", "input_shapes": [(1, 2), (1, 2)], "shape": (1, 2)},
        {"name": "relu", "input_shapes": [(1, 2)], "shape": (1, 2)},
        {"name": "output", "shape": (1, 2)},
    ],
    "edge_list": [(0, 2), (1, 2), (2, 3), (3, 4)],
    "index_map": {(0, 2): 0, (1, 2): 1},
}


# %%
sample_graph = graph.Graph(**sample_graph_spec)
sample_graph.show()

# %%
compiled = graph.CompiledGraph.from_graph(sample_graph)

# %%
compiled([torch.tensor([-4, 2]), torch.tensor([2, 4])])
