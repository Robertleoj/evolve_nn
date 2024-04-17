import random
from copy import deepcopy

from project.evolution.individual import Individual
from project.graph.graph import Graph
from project.type_defs import EvolutionConfig, GraphMutHP, TrainingHP
from project.variation_ops.graph_mutation import mutate_graph


def mutate_individual(individual: Individual, evolution_config: EvolutionConfig) -> Individual:
    """Mutate an individual."""
    graph_mut_hps = mutate_graph_hps(individual.graph_mut_hps, evolution_config)
    graph = mutate_graph(individual.graph, graph_mut_hps)
    training_hp = mutate_training_hp(individual.training_hp)

    return Individual(graph_mut_hps=graph_mut_hps, graph=graph, training_hp=training_hp)


def mutate_training_hp(training_hp: TrainingHP) -> TrainingHP:
    """Mutate training hyperparameters."""
    return TrainingHP(
        lr=training_hp.lr * random.uniform(0.9, 1.1), momentum=training_hp.momentum * random.uniform(0.9, 1.1)
    )


def mutate_graph_hps(hp: GraphMutHP, evolution_config: EvolutionConfig) -> GraphMutHP:
    new_hp = deepcopy(hp)

    if evolution_config.mutate_num_mutations:
        new_hp.max_num_mutations = max(hp.max_num_mutations + random.choice([-1, 0, 1]), 1)
    else:
        new_hp.max_num_mutations = hp.max_num_mutations

    new_mut_probs = {}
    for k, v in hp.mutation_probabilities.items():
        new_mut_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.mutation_probabilities = new_mut_probs

    new_sub_mut_probs = {}
    for k, v in hp.subgraph_mutation_probabilities.items():
        new_sub_mut_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.subgraph_mutation_probabilities = new_sub_mut_probs

    new_operator_probs = {}
    for k, v in hp.operator_probabilities.items():
        new_operator_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.operator_probabilities = new_operator_probs

    new_sub_op_probs = {}
    for k, v in hp.subgraph_operator_probabilities.items():
        new_sub_op_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.subgraph_operator_probabilities = new_sub_op_probs

    return new_hp


def recombine_training_hps(training_hp1: TrainingHP, training_hp2: TrainingHP) -> TrainingHP:
    """Recombine training hyperparameters."""
    return TrainingHP(
        lr=random.choice([training_hp1.lr, training_hp2.lr]),
        momentum=random.choice([training_hp1.momentum, training_hp2.momentum]),
    )


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


def recombine_individuals(
    individual1: Individual, individual2: Individual, evolution_config: EvolutionConfig
) -> Individual:
    """Recombine two individuals."""
    graph_mut_hps = recombine_graph_hps(
        individual1.graph_mut_hps, individual2.graph_mut_hps, evolution_config=evolution_config
    )
    graph = recombine_graphs(individual1.graph, individual2.graph, graph_mut_hps)
    training_hp = recombine_training_hps(individual1.training_hp, individual2.training_hp)

    return Individual(graph_mut_hps=graph_mut_hps, graph=graph, training_hp=training_hp)


__all__ = [
    "mutate_individual",
    "mutate_training_hp",
    "mutate_graph_hps",
    "recombine_training_hps",
    "recombine_graphs",
    "recombine_graph_hps",
    "recombine_individuals",
    "mutate_graph",
    "mutate_graph_hps",
]
