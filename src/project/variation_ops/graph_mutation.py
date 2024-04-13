"""Mutation operators for graphs."""
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import networkx as nx
from project.graph.graph import Graph, show_graph, ParameterNode, op_node_name_to_node, OperatorNode, node_name_to_node, Node
from project.type_defs import EvolutionConfig, GraphMutHP


def check_graph_validity(graph: Graph) -> tuple[bool, str]:
    """Check that the graph is a valid computation graph.

    TODO:
    * Check that all subgraphs are valid
    * Check that num_inputs to subgraphs agree with the number of incoming edges
    """

    # the graph must be weakly connected
    g_nx = graph.get_nx()
    if not nx.is_weakly_connected(g_nx):
        return False, "Graph is not weakly connected"

    # must be acyclic
    if not nx.is_directed_acyclic_graph(g_nx):
        return False, "Graph is not acyclic"

    # All nodes in the graph must be node instances
    for node_id, node in graph.id_to_node.items():
        if not isinstance(node, Node):
            return False, f"Node {node_id} is not an instance of Node"

    adj_list = graph.adj_list
    rev_adj_list = graph.rev_adj_list
    input_nodes = graph.input_nodes()

    # must have at least one input node
    if len(input_nodes) == 0:
        return False, "Graph has no input nodes"

    # input nodes must have no incoming edges
    for node_id in input_nodes:
        if not len(rev_adj_list[node_id]) == 0:
            return False, f"Input node {node_id} has incoming edges"

    # must have at least one output node
    output_nodes = graph.output_nodes()
    if len(output_nodes) == 0:
        return False, "Graph has no output nodes"

    for node_id in output_nodes:
        # output nodes must have no outgoing edges
        if not len(adj_list[node_id]) == 0:
            return False, f"Output node {node_id} has outgoing edges"

        # output nodes must have exactly one incoming edge
        if not len(rev_adj_list[node_id]) == 1:
            return False, f"Data node {node_id} does not have exactly one incoming edge"


    for node_id in graph.operator_nodes():
        # Operator nodes have at least one outgoing edge
        if not len(adj_list[node_id]) > 0:
            return False, f"Operator node {node_id} does not have outgoing edges"

        # Operator nodes have number of input nodes that is within the bounds
        node = graph.id_to_node[node_id]
        assert isinstance(node, OperatorNode)

        lower_bound, upper_bound = node.n_inputs
        n_inputs = len(rev_adj_list[node_id])
        if n_inputs < lower_bound or (upper_bound is not None and n_inputs > upper_bound):
            return False, f"Operator node {node_id} has {n_inputs} inputs, expected range {lower_bound} - {upper_bound}"

    # 8: Parameter nodes only point to operator nodes
    for node_id in graph.parameter_nodes():

        if not len(adj_list[node_id]) > 0:
            return False, f"Parameter node {node_id} does not point to any operator nodes"

        for to_node_id in adj_list[node_id]:
            to_node = graph.id_to_node[to_node_id]
            if not isinstance(
                to_node, OperatorNode
            ):
                return False, f"Parameter node {node_id} points to a non-operator node {to_node_id}"

    return True, ""


def available_opnodes(graph: Graph, no_single_params: bool = False) -> list[str]:
    opnodes = graph.operator_nodes()
    rev_adj_list = graph.rev_adj_list
    forward_adj_list = graph.adj_list

    available_nodes = []

    for node_id in opnodes:
        inputs = rev_adj_list[node_id]
        if no_single_params:
            found_single_param = False
            for i in inputs:
                if isinstance(graph.id_to_node[i], ParameterNode):
                    if len(set(forward_adj_list[i])) == 1:
                        found_single_param = True
                        break
            if found_single_param:
                continue

        node = graph.id_to_node[node_id]
        
        print(node)
        assert isinstance(node, OperatorNode)
        lower_bound, upper_bound = node.n_inputs

        if upper_bound is None:
            # no upper bound
            available_nodes.append(node_id)
            continue

        n_inputs = len(rev_adj_list[node_id])

        if n_inputs < upper_bound:
            available_nodes.append(node_id)

    return available_nodes


def expand_edge(graph: Graph, mutation_hps: GraphMutHP) -> tuple[Graph, bool]:
    graph = deepcopy(graph)

    # get a random edge
    random_edge = random.choice(graph.edge_list)

    node1 = random_edge[0]
    node2 = random_edge[1]

    # delete the edge
    graph.delete_edge(*random_edge)

    exclude_ops = ["graph"]

    op_names = [op for op in op_node_name_to_node.keys() if op not in exclude_ops]
    op_probs = [mutation_hps.operator_probabilities[op] for op in op_names]

    # do not use the 'graph' operator

    # select a random operator
    op_name_to_use = random.choices(op_names, weights=op_probs)[0]

    new_node = node_name_to_node[op_name_to_use]()

    # create a new node
    node_id = graph.add_node(new_node)

    # add the new edges
    graph.add_edges([(node1, node_id), (node_id, node2)])

    return graph, True


def add_parameter(graph: Graph, mutation_hps: GraphMutHP) -> tuple[Graph, bool]:
    available_nodes = available_opnodes(graph, no_single_params=True)

    if len(available_nodes) == 0:
        return graph, False

    graph = deepcopy(graph)

    # create a new node
    parameter_node = ParameterNode()

    node_id = graph.add_node(parameter_node)

    random_node = random.choice(available_nodes)

    graph.add_edge(node_id, random_node)

    return graph, True


def add_edge(graph: Graph, mutation_hps, tries: int = 30) -> tuple[Graph, bool]:
    output_nodes = set(graph.output_nodes())

    node_1_candidates = [
        node_id for node_id in graph.id_to_node 
        if node_id not in output_nodes
    ]

    node_2_candidates = available_opnodes(graph)

    if len(node_1_candidates) == 0 or len(node_2_candidates) == 0:
        return graph, False

    g_nx = graph.get_nx()
    for _ in range(tries):
        # get a random edge
        node1 = random.choice(node_1_candidates)
        node2 = random.choice(node_2_candidates)

        if nx.has_path(g_nx, node2, node1):
            continue

        graph = deepcopy(graph)
        graph.add_edge(node1, node2)

        return graph, True

    return graph, False


def delete_edge(graph: Graph, mutation_hps, tries=100) -> tuple[Graph, bool]:
    g_nx = graph.get_nx()

    adj_list = graph.adj_list
    rev_adj_list = graph.rev_adj_list

    for _ in range(tries):
        node1_id, node2_id = random.choice(graph.edge_list)

        # 1: the edge must not be the only incoming edge
        if len(rev_adj_list[node2_id]) == 1:
            continue

        # 2: the edge must not be the only outgoing edge
        if len(adj_list[node1_id]) == 1:
            continue

        # 3: We cannot disconnect the graph by removing the edge
        g_nx.remove_edge(node1_id, node2_id)
        if not nx.is_weakly_connected(g_nx):
            g_nx.add_edge(node1_id, node2_id)
            continue

        graph = deepcopy(graph)
        graph.delete_edge(node1_id, node2_id)
        return graph, True

    return graph, False


def delete_parameter(graph: Graph, mutation_hps: GraphMutHP, tries=30) -> tuple[Graph, bool]:
    paramnodes: list[str] = [
        node_id for node_id, node in graph.id_to_node.items()
        if isinstance(node, ParameterNode)
    ]

    if len(paramnodes) == 0:
        return graph, False

    adj_list = graph.adj_list
    rev_adj_list = graph.rev_adj_list
    for _ in range(tries):
        random_parameter = random.choice(paramnodes)

        out_neighbors = adj_list[random_parameter]

        can_delete = True
        for out_neighbor in out_neighbors:
            out_neighbor_in_neighbors = rev_adj_list[out_neighbor]

            if len(set(out_neighbor_in_neighbors)) == 1:
                can_delete = False
        if not can_delete:
            continue

        graph_copy = deepcopy(graph)

        graph_copy.delete_node(random_parameter)

        return graph_copy, True

    return graph, False


def delete_operator(graph: Graph, mutation_hps, tries=30) -> tuple[Graph, bool]:
    opnodes = graph.operator_nodes()

    if len(opnodes) == 0:
        return graph, False

    adj_list = graph.adj_list
    rev_adj_list = graph.rev_adj_list

    for _ in range(tries):
        g_nx = graph.get_nx()

        random_operator = random.choice(opnodes)

        outgoing_nodes = list(set(adj_list[random_operator]))
        incoming_nodes = list(set(rev_adj_list[random_operator]))

        random.shuffle(outgoing_nodes)
        random.shuffle(incoming_nodes)

        # randomly assign the incoming edges to the outgoing edges
        edges_added = []

        for i, o in zip(incoming_nodes, outgoing_nodes):
            edges_added.append((i, o))

        if len(outgoing_nodes) > len(incoming_nodes):
            for o in outgoing_nodes[len(incoming_nodes) :]:
                edges_added.append((random.choice(incoming_nodes), o))

        for edge in edges_added:
            g_nx.add_edge(*edge)

        # delete the operator
        g_nx.remove_node(random_operator)

        if not nx.is_weakly_connected(g_nx):
            continue

        # graph is weakly connected - we can go ahead and remove the node
        graph_cpy = deepcopy(graph)

        graph_cpy.delete_node(random_operator)
        graph_cpy.add_edges(edges_added)

        # check if the graph is still valid
        all_ok, _ = check_graph_validity(graph_cpy)

        if not all_ok:
            continue

        return graph_cpy, True

    return graph, False


graph_mutation_functions: dict[str, Callable[[Graph, GraphMutHP], tuple[Graph, bool]]] = {
    "expand_edge": expand_edge,
    "add_edge": add_edge,
    "add_parameter": add_parameter,
    "delete_operator": delete_operator,
    "delete_edge": delete_edge,
    "delete_parameter": delete_parameter,
}



def mutate_graph(graph: Graph, mutation_hyperparams: GraphMutHP) -> Graph:
    mutated = graph

    max_num_mutations = mutation_hyperparams.max_num_mutations
    names, probs = zip(*list(mutation_hyperparams.mutation_probabilities.items()))

    mutations_performed = 0
    num_mutations = random.randint(1, max_num_mutations)
    while mutations_performed < num_mutations:

        mutation_name = random.choices(names, weights=probs)[0]
        mutation_function = graph_mutation_functions[mutation_name]
        mutated, changed = mutation_function(mutated, mutation_hyperparams)

        valid, reason = check_graph_validity(mutated)
        if not valid:
            print(f"Tried to mutate with {mutation_name}, but graph is invalid: {reason}")
            show_graph(mutated)
            raise RuntimeError(f"Invalid graph after mutation with {mutation_name}")

        if changed:
            mutations_performed += 1

    return mutated


def mutate_graph_hps(hp: GraphMutHP, evolution_config: EvolutionConfig) -> GraphMutHP:
    new_hp = deepcopy(hp)

    if evolution_config.mutate_num_mutations:
        new_hp.max_num_mutations = max(hp.max_num_mutations + random.choice([-1, 0, 1]), 1)
    else:
        new_hp.max_num_mutations = hp.max_num_mutations

    new_probs = {}
    for k, v in hp.mutation_probabilities.items():
        new_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.mutation_probabilities = new_probs

    new_operator_probs = {}
    for k, v in hp.operator_probabilities.items():
        new_operator_probs[k] = v * random.uniform(0.8, 1.2)

    return new_hp

