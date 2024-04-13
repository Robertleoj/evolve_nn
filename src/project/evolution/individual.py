from dataclasses import dataclass

from project.graph.graph import Graph
from project.type_defs import TrainingHP
from project.variation_ops.graph_mutation import (
    GraphMutHP,
)


@dataclass
class Individual:
    graph: Graph
    graph_mut_hps: GraphMutHP
    training_hp: TrainingHP
