from dataclasses import dataclass

from compute_graph.graph.graph import Graph
from compute_graph.type_defs import GraphMutHP, TrainingHP


@dataclass
class Individual:
    graph: Graph
    graph_mut_hps: GraphMutHP
    training_hp: TrainingHP
