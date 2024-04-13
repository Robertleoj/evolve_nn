import random

from project.evolution.individual import Individual
from project.graph.graph import make_graph, op_node_name_to_node
from project.type_defs import EvolutionConfig, GraphMutHP, TrainingHP
from project.variation_ops.graph_mutation import graph_mutation_functions, mutate_graph


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


def random_training_hp() -> TrainingHP:
    """Generate random training hyperparameters."""
    return TrainingHP(lr=random.uniform(1e-5, 1e-3), momentum=random.uniform(0.1, 0.9))


def random_individual(init_graph_spec: dict, evolution_config: EvolutionConfig) -> Individual:
    init_hps = random_graph_mut_hps(evolution_config)
    init_graph = mutate_graph(make_graph(**init_graph_spec), init_hps)
    init_training_hp = random_training_hp()

    return Individual(graph_mut_hps=init_hps, graph=init_graph, training_hp=init_training_hp)
