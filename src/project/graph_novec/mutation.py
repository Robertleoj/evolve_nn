"""Mutation operators for graphs"""
from project.graph_novec.graph import Graph
from project.graph_novec.nodes import DataNode, OperatorNode, operator_nodes, node_from_name
import random
import networkx as nx
from copy import deepcopy


def expand_edge(graph: Graph) -> tuple[Graph, bool]:

    graph = deepcopy(graph)

    # get a random edge
    random_edge = random.choice(graph.edge_list)

    node1 = random_edge[0]
    node2 = random_edge[1]

    # delete the edge
    graph.edge_list.remove(random_edge)

    # select a random operator
    op_to_use = random.choice(operator_nodes)

    # create a new node
    new_node = op_to_use()
    node_id = graph.add_node(new_node)

    # add the new edges
    graph.edge_list.extend([
        (node1, node_id),
        (node_id, node2)
    ])

    return graph, True


def add_edge(graph: Graph, tries: int = 30) -> tuple[Graph, bool]:
    g_nx = graph.get_nx()

    opnodes = graph.operator_nodes()
    output_nodes = set(graph.output_nodes())

    node_1_candidates = [
        node_id for node_id in graph.node_ids
        if node_id not in output_nodes
    ]

    for _ in range(tries):
        # get a random edge
        node1 = random.choice(node_1_candidates)
        node2 = random.choice(opnodes)

        if nx.has_path(g_nx, node2, node1):
            continue

        graph = deepcopy(graph)
        graph.edge_list.append((node1, node2))

        return graph, True

    return graph, False


def expand_operator(graph: Graph) -> tuple[Graph, bool]:
    graph = deepcopy(graph)

    
    

