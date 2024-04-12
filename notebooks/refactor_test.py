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
from project.graph.graph import Graph, make_graph, show_graph

# %%
node_specs = [
    {
        "name": "input"
    },
    {
        "name": "input"
    },
    {
        "name": "add"
    },
    {
        "name": "output"
    }
]

# %%
rev_adj_list = [
    [],
    [],
    [0, 1],
    [2]
]

# %%
graph = make_graph(node_specs, rev_adj_list)

# %%
show_graph(graph)
