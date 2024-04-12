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
from project.graph.graph import Graph, make_graph, show_graph, CompiledGraph
import torch

# %%
graph_spec = {
    "subgraph_specs": [
        {
            "node_specs": [
                {"name": "input"},
                {"name": "input"},
                {"name": "add"},
                {"name": "output"}
            ],
            "rev_adj_list": [
                [], [], [0, 1], [2]
            ],
            "input_node_order": [0, 1]
        }
    ],
    "node_specs": [
        {
            "name": "input"
        },
        {
            "name": "input"
        },
        {
            "name": "graph",
            "subgraph_idx": 0
        },
        {
            "name": "graph",
            "subgraph_idx": 0
        },
        {
            "name": "output"
        }
    ],
    "rev_adj_list":  [
        [],
        [],
        [0, 1],
        [2, 2],
        [3]
    ],
    "input_node_order": [0, 1],
    "output_node_order": [4]
}
    

# %%
g = make_graph(**graph_spec)

# %%
g.subgraphs

# %%
show_graph(g)

# %%
compiled = CompiledGraph.from_graph(g)

# %%
compiled([torch.tensor([1, 2]), torch.tensor([4, 5])])
