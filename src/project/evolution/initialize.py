import random

import project.graph.graph as graph_
import project.graph.nodes as nodes_
from project.evolution.individual import Individual
from project.type_defs import EvolutionConfig, GraphMutHP, TrainingHP
from project.variation_ops.graph_mutation import graph_mutation_functions, mutate_graph


def random_graph_mut_hps(evolution_config: EvolutionConfig) -> GraphMutHP:
    mut_probs = {}
    sub_mut_probs = {}
    for k in graph_mutation_functions.keys():
        mut_probs[k] = random.uniform(0, 1)
        sub_mut_probs[k] = random.uniform(0, 1)

    if evolution_config.mutate_num_mutations:
        max_num_mutations = random.randint(1, 10)
    else:
        assert evolution_config.max_num_mutations is not None
        max_num_mutations = evolution_config.max_num_mutations

    sub_op_probs = {}
    operator_probabilities = {}
    for op in nodes_.op_node_name_to_node.keys():
        sub_op_probs[op] = random.uniform(0, 1)
        operator_probabilities[op] = random.uniform(0, 1)

    return GraphMutHP(
        mutation_probabilities=mut_probs,
        subgraph_mutation_probabilities=sub_mut_probs,
        max_num_mutations=max_num_mutations,
        operator_probabilities=operator_probabilities,
        subgraph_operator_probabilities=sub_op_probs,
        max_num_subgraphs=evolution_config.max_num_subgraphs,
        max_subgraph_depth=evolution_config.max_subgraph_depth,
    )


def random_training_hp() -> TrainingHP:
    """Generate random training hyperparameters."""
    return TrainingHP(lr=random.uniform(1e-5, 1e-3), momentum=random.uniform(0.1, 0.9))


def random_individual(init_graph_spec: dict, evolution_config: EvolutionConfig) -> Individual:
    init_hps = random_graph_mut_hps(evolution_config)
    init_graph = mutate_graph(graph_.make_graph(**init_graph_spec), init_hps)
    init_training_hp = random_training_hp()

    return Individual(graph_mut_hps=init_hps, graph=init_graph, training_hp=init_training_hp)


def initialize_population(init_spec: dict, evolution_config: EvolutionConfig) -> list[Individual]:
    population = [random_individual(init_spec, evolution_config) for _ in range(evolution_config.population_size)]
    return population
