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
import torch
from project.graph.graph import CompiledGraph, make_graph, show_graph, SubGraphNode, OperatorNode
from project.variation_ops.graph_mutation import (
    expand_edge,
    add_parameter,
    add_edge,
    delete_edge,
    delete_parameter,
    delete_operator,
    mutate_graph,
)
from project.type_defs import GraphMutHP, EvolutionConfig
from project.evolution.initialize import random_graph_mut_hps

# %%
graph_spec = {
    "subgraph_specs": [
        {
            "node_specs": [{"name": "input"}, {"name": "input"}, {"name": "add"}, {"name": "output"}],
            "rev_adj_list": [[], [], [0, 1], [2]],
            "input_node_order": [0, 1],
        }
    ],
    "node_specs": [
        {"name": "input"},
        {"name": "input"},
        {"name": "graph", "subgraph_idx": 0},
        {"name": "graph", "subgraph_idx": 0},
        {"name": "output"},
    ],
    "rev_adj_list": [[], [], [0, 1], [2, 2], [3]],
    "input_node_order": [0, 1],
    "output_node_order": [4],
}


# %%
g = make_graph(**graph_spec)

# %%
g.subgraphs

# %%
show_graph(g)

# %%
g.id_to_node

# %%
n = next(n for n_id, n in g.id_to_node.items() if isinstance(n, SubGraphNode))

# %%
type(n)

# %%
isinstance(n, OperatorNode)

# %%
compiled = CompiledGraph.from_graph(g)

# %%
compiled([torch.tensor([1, 2]), torch.tensor([4, 5])])

# %%
ev_config = EvolutionConfig()

# %%
from project.graph.graph import show_compiled


mutated = g
graph_mut_hps = random_graph_mut_hps(ev_config)
for _ in range(20):
    try:
        mutated = mutate_graph(mutated, graph_mut_hps)
    except Exception as e:
        show_graph(mutated)
        print(mutated.rev_adj_list)
        print(mutated.id_to_node)
        raise e

show_graph(mutated, show_node_ids=False)
mut_compiled = CompiledGraph.from_graph(mutated)
# show_compiled(mut_compiled)
display(mut_compiled([torch.tensor([1.0, 2.0]), torch.tensor([4.0, 5.0])]))
# mutated, changed = expand_edge(mutated, graph_mut_hps)
# mutated, changed = add_parameter(mutated, graph_mut_hps)
# mutated, changed = add_edge(mutated, graph_mut_hps)
