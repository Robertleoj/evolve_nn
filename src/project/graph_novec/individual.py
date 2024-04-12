from dataclasses import dataclass
import random
from project.graph_novec.graph import Graph, make_graph
from project.graph_novec.graph_mutation import mutate_graph, GraphMutHP, mutate_graph_hps, recombine_graph_hps, recombine_graphs, random_graph_mut_hps
from project.type_defs import EvolutionConfig


@dataclass
class TrainingHP:
    """Training hyperparameters."""
    lr: float
    momentum: float

def mutate_training_hp(training_hp: TrainingHP):
    """Mutate training hyperparameters."""
    return TrainingHP(
        lr=training_hp.lr * random.uniform(0.9, 1.1),
        momentum=training_hp.momentum * random.uniform(0.9, 1.1)
    )

def recombine_training_hps(training_hp1: TrainingHP, training_hp2: TrainingHP):
    """Recombine training hyperparameters."""
    return TrainingHP(
        lr=random.choice([training_hp1.lr, training_hp2.lr]),
        momentum=random.choice([training_hp1.momentum, training_hp2.momentum])
    )

def random_training_hp():
    """Generate random training hyperparameters."""
    return TrainingHP(
        lr=random.uniform(1e-5, 1e-3),
        momentum=random.uniform(0.1, 0.9)
    )

@dataclass
class Individual:
    graph_mut_hps: GraphMutHP
    graph: Graph
    training_hp: TrainingHP


def mutate_individual(individual: Individual, evolution_config: EvolutionConfig):
    """Mutate an individual."""
    graph_mut_hps = mutate_graph_hps(
        individual.graph_mut_hps, 
        evolution_config
    )
    graph = mutate_graph(individual.graph, graph_mut_hps)
    training_hp = mutate_training_hp(individual.training_hp)
    return Individual(graph_mut_hps, graph, training_hp)


def recombine_individuals(individual1: Individual, individual2: Individual, evolution_config: EvolutionConfig):
    """Recombine two individuals."""
    graph_mut_hps = recombine_graph_hps(individual1.graph_mut_hps, individual2.graph_mut_hps, evolution_config=evolution_config)
    graph = recombine_graphs(individual1.graph, individual2.graph, graph_mut_hps)
    training_hp = recombine_training_hps(individual1.training_hp, individual2.training_hp)
    return Individual(graph_mut_hps, graph, training_hp)
    

def random_individual(init_graph_spec: dict, evolution_config: EvolutionConfig) -> Individual:
    init_hps = random_graph_mut_hps(evolution_config)
    init_graph = mutate_graph(make_graph(**init_graph_spec), init_hps)
    init_training_hp = random_training_hp()

    return Individual(
        graph_mut_hps=init_hps,
        graph=init_graph,
        training_hp=init_training_hp
    )