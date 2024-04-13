from copy import deepcopy
from project.type_defs import GraphMutHP, EvolutionConfig
from project.graph.graph import Graph


def recombine_graphs(graph1: Graph, graph2: Graph, hp: GraphMutHP) -> Graph:
    return graph1


def recombine_graph_hps(hp1: GraphMutHP, hp2: GraphMutHP, evolution_config: EvolutionConfig) -> GraphMutHP:
    new_hp = deepcopy(hp1)

    if evolution_config.mutate_num_mutations:
        new_hp.max_num_mutations = max(int((hp1.max_num_mutations + hp2.max_num_mutations) / 2), 1)
    else:
        new_hp.max_num_mutations = hp1.max_num_mutations

    new_probs = {}
    for k in hp1.mutation_probabilities.keys():
        new_probs[k] = (hp1.mutation_probabilities[k] + hp2.mutation_probabilities[k]) / 2

    new_hp.mutation_probabilities = new_probs

    new_op_probs = {}
    for k in hp1.operator_probabilities.keys():
        new_op_probs[k] = (hp1.operator_probabilities[k] + hp2.operator_probabilities[k]) / 2

    new_hp.operator_probabilities = new_op_probs
    return new_hp


