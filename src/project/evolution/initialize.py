from project.type_defs import EvolutionConfig, GraphMutHP
from project.variation_ops.graph_mutation import graph_mutation_functions
from project.graph.graph import op_node_name_to_node

import random


def random_graph_mut_hps(evolution_config: EvolutionConfig) -> GraphMutHP:
    probs = {}
    for k in graph_mutation_functions.keys():
        probs[k] = random.uniform(0, 1)

    if evolution_config.mutate_num_mutations:
        max_num_mutations = random.randint(1, 10)
    else:
        assert evolution_config.max_num_mutations is not None
        max_num_mutations = evolution_config.max_num_mutations

    operator_probabilities = {op: random.uniform(0, 1) for op in op_node_name_to_node.keys()}

    return GraphMutHP(
        mutation_probabilities=probs, max_num_mutations=max_num_mutations, operator_probabilities=operator_probabilities
    )

