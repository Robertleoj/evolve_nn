"""Mutation operators for graphs."""
import random
from collections import defaultdict
from copy import deepcopy
from queue import deque
from typing import Callable

import networkx as nx

import project.graph.graph as graph_
import project.graph.nodes as nodes_
from project.type_defs import GraphMutHP
from project.utils.graph_utils import are_all_reachable
from project.utils.rand_utils import weighted_random

DEFAULT_NUM_TRIES = 5

MutFunction = Callable[[graph_.Graph, GraphMutHP, int], tuple[graph_.Graph, bool]]


def get_reverse_lengths(graph: graph_.Graph) -> dict:
    """Using BFS from input nodes, get the number of reverse edges needed to reach each node."""
    # Reverse distances dictionary initialized to infinity
    rev_distances: defaultdict[str, float] = defaultdict(lambda: float("inf"))

    # Initializing deque with input nodes and distance 0
    start_nodes = graph.input_nodes() + graph.response_nodes()
    queue = deque([(node_id, 0) for node_id in start_nodes])

    while queue:
        current_id, current_distance = queue.popleft()
        # Update the reverse distance if a shorter path is found
        if current_distance < rev_distances[current_id]:
            rev_distances[current_id] = current_distance
            # Traverse through all adjacent nodes (forward and reverse)
            # Forward edges: distance doesn't increase
            for neighbor_id in graph.adj_list.get(current_id, []):
                queue.append((neighbor_id, current_distance))
            # Reverse edges: increment distance by 1
            for neighbor_id in graph.rev_adj_list.get(current_id, []):
                queue.append((neighbor_id, current_distance + 1))

    # Fill distances for nodes that weren't reachable
    for node_id in graph.id_to_node:
        if node_id not in rev_distances:
            rev_distances[node_id] = float("inf")

    return dict(rev_distances)


def check_graph_validity(graph: graph_.Graph) -> tuple[bool, str]:
    """Check that the graph is a valid computation graph.

    Todo:
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
        if not isinstance(node, nodes_.Node):
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
        if graph.is_subgraph:
            # output nodes must have no outgoing edges
            if not len(adj_list[node_id]) == 0:
                return False, f"Output node {node_id} has outgoing edges"

        # output nodes must have exactly one incoming edge
        if not len(rev_adj_list[node_id]) == 1:
            return False, f"Data node {node_id} does not have exactly one incoming edge"

    opnodes = graph.operator_nodes()
    for node_id in opnodes:
        # Operator nodes have at least one outgoing edge
        if not len(adj_list[node_id]) > 0:
            return False, f"Operator node {node_id} does not have outgoing edges"

        # Operator nodes have number of input nodes that is within the bounds
        node = graph.id_to_node[node_id]
        assert isinstance(node, nodes_.OperatorNode)

        lower_bound, upper_bound = node.n_inputs
        n_inputs = len(rev_adj_list[node_id])
        if n_inputs < lower_bound or (upper_bound is not None and n_inputs > upper_bound):
            return False, f"Operator node {node_id} has {n_inputs} inputs, expected range {lower_bound} - {upper_bound}"

    # Parameter nodes only point to operator nodes
    for node_id in graph.parameter_nodes():
        if not len(adj_list[node_id]) > 0:
            return False, f"Parameter node {node_id} does not point to any operator nodes"

        for to_node_id in adj_list[node_id]:
            to_node = graph.id_to_node[to_node_id]
            if not isinstance(to_node, nodes_.OperatorNode):
                return False, f"Parameter node {node_id} points to a non-operator node {to_node_id}"

    # operator nodes cannot have multiple parameters that only point to them
    for node_id in opnodes:
        node = graph.id_to_node[node_id]
        inputs = rev_adj_list[node_id]

        param_inputs = [i for i in inputs if isinstance(graph.id_to_node[i], nodes_.ParameterNode)]

        has_only_single_output = {}
        for param_id in param_inputs:
            outputs = graph.adj_list[param_id]
            if len(set(outputs)) == 1:
                has_only_single_output[param_id] = len(outputs)

        if len(has_only_single_output) > 1:
            return False, f"Operator node {node_id} has multiple parameters with only one output"

    # all output nodes must be reachable from some input node
    if not are_all_reachable(g_nx, set(input_nodes), set(output_nodes)):
        return False, "Not all output nodes are reachable from input nodes"

    if not graph.is_subgraph:
        # response nodes cannot have paths to the output
        for node_id in graph.response_nodes():
            for output_node in graph.output_nodes():
                if nx.has_path(g_nx, node_id, output_node):
                    return False, f"Response node {node_id} has a path to the output"

        # loss output cannot have outputs
        if graph.loss_output_node is not None:
            loss_out = graph.loss_output_node
            assert loss_out is not None

            # all parameter and subgraph nodes must have paths to the loss node, if exists
            for node_id in graph.parameter_nodes() + graph.subgraph_nodes():
                if not nx.has_path(g_nx, node_id, loss_out):
                    return False, f"Parameter node {node_id} has no path to loss output node"

    # no more than 1 reverse distance from input nodes
    rev_distances = get_reverse_lengths(graph)
    for node_id, distance in rev_distances.items():
        if distance > 1:
            return False, f"Node {node_id} has reverse distance {distance} from input nodes"

    return True, ""


def available_opnodes(graph: graph_.Graph, no_single_params: bool = False) -> list[str]:
    opnodes = graph.operator_nodes()
    rev_adj_list = graph.rev_adj_list
    forward_adj_list = graph.adj_list

    available_nodes = []

    for node_id in opnodes:
        inputs = rev_adj_list[node_id]
        if no_single_params:
            found_single_param = False
            for i in inputs:
                if isinstance(graph.id_to_node[i], nodes_.ParameterNode):
                    if len(set(forward_adj_list[i])) == 1:
                        found_single_param = True
                        break
            if found_single_param:
                continue

        node = graph.id_to_node[node_id]

        assert isinstance(node, nodes_.OperatorNode)
        lower_bound, upper_bound = node.n_inputs

        if upper_bound is None:
            # no upper bound
            available_nodes.append(node_id)
            continue

        n_inputs = len(rev_adj_list[node_id])

        if n_inputs < upper_bound:
            available_nodes.append(node_id)

    return available_nodes


def choose_op_to_add(graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0) -> nodes_.OperatorNode:
    exclude_ops: list[str] = []

    if len(graph.subgraphs) == 0:
        exclude_ops.append("graph")

    op_names = [op for op in nodes_.op_node_name_to_node.keys() if op not in exclude_ops]
    if graph.is_subgraph:
        op_prob_dict = mutation_hps.subgraph_operator_probabilities
    else:
        op_prob_dict = mutation_hps.operator_probabilities

    op_probs = [op_prob_dict[op] for op in op_names]

    # select a random operator
    op_name_to_use = random.choices(op_names, weights=op_probs)[0]

    if op_name_to_use == "graph":
        subgraph = random.choice(graph.subgraphs)
        subgraph_node = nodes_.SubGraphNode(subgraph)
        return subgraph_node

    return nodes_.op_node_name_to_node[op_name_to_use]()


def nodes_that_can_have_outputs(graph: graph_.Graph) -> list[str]:
    candidates = set(graph.id_to_node.keys())
    if graph.is_subgraph:
        candidates -= set(graph.output_nodes())

    return list(candidates)


def expand_edge(
    graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0, tries=DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    for _ in range(tries):
        graph_cpy = deepcopy(graph)

        # get a random edge
        random_edge = random.choice(graph_cpy.edge_list)

        node1 = random_edge[0]
        node2 = random_edge[1]

        # delete the edge
        graph_cpy.delete_edge(*random_edge)

        new_node = choose_op_to_add(graph_cpy, mutation_hps, subgraph_depth=subgraph_depth)

        # create a new node
        node_id = graph_cpy.add_node(new_node)

        node_min_num_inputs = new_node.n_inputs[0]

        graph_cpy.get_nx()

        node_choices = nodes_that_can_have_outputs(graph_cpy)
        edges_to_add = [(node1, node_id), (node_id, node2)]
        graph_cpy.add_edges(edges_to_add)
        g_nx = graph_cpy.get_nx()

        # add new edges until we ahve minimum number of inputs, keeping graph acyclic
        curr_num_inputs = 1
        while curr_num_inputs < node_min_num_inputs:
            random_input_node = random.choice(node_choices)
            g_nx.add_edge(random_input_node, node_id)
            if not nx.is_directed_acyclic_graph(g_nx):
                g_nx.remove_edge(random_input_node, node_id)
                continue
            graph_cpy.add_edge(random_input_node, node_id)
            curr_num_inputs += 1

        if not check_graph_validity(graph_cpy)[0]:
            continue

        return graph_cpy, True

    return graph, False


def add_parameter(
    graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0, tries=DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    available_nodes = available_opnodes(graph, no_single_params=True)

    if len(available_nodes) == 0:
        return graph, False

    for _ in range(tries):
        graph_cpy = deepcopy(graph)

        # create a new node
        parameter_node = nodes_.ParameterNode()

        node_id = graph_cpy.add_node(parameter_node)

        random_node = random.choice(available_nodes)

        graph_cpy.add_edge(node_id, random_node)

        if not check_graph_validity(graph_cpy)[0]:
            continue

        return graph_cpy, True

    return graph, False


def add_edge(
    graph: graph_.Graph, mutation_hps, subgraph_depth: int = 0, tries: int = DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    node_1_candidates = nodes_that_can_have_outputs(graph)
    node_2_candidates = available_opnodes(graph)

    if len(node_1_candidates) == 0 or len(node_2_candidates) == 0:
        return graph, False

    for _ in range(tries):
        # get a random edge
        node1 = random.choice(node_1_candidates)
        node2 = random.choice(node_2_candidates)

        graph_cpy = deepcopy(graph)
        graph_cpy.add_edge(node1, node2)

        if not check_graph_validity(graph_cpy)[0]:
            continue

        return graph_cpy, True

    return graph, False


def delete_edge(
    graph: graph_.Graph, mutation_hps, subgraph_depth: int = 0, tries=DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    for _ in range(tries):
        node1_id, node2_id = random.choice(graph.edge_list)

        graph_cpy = deepcopy(graph)
        graph_cpy.delete_edge(node1_id, node2_id)

        if not check_graph_validity(graph_cpy)[0]:
            continue

        return graph_cpy, True

    return graph, False


def delete_parameter(
    graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0, tries=DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    paramnodes: list[str] = [
        node_id for node_id, node in graph.id_to_node.items() if isinstance(node, nodes_.ParameterNode)
    ]

    if len(paramnodes) == 0:
        return graph, False

    for _ in range(tries):
        random_parameter = random.choice(paramnodes)

        graph_copy = deepcopy(graph)

        graph_copy.delete_node(random_parameter)

        if not check_graph_validity(graph_copy)[0]:
            continue

        return graph_copy, True

    return graph, False


def delete_operator(
    graph: graph_.Graph, mutation_hps, subgraph_depth: int = 0, tries=DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    opnodes = graph.operator_nodes()

    if len(opnodes) == 0:
        return graph, False

    adj_list = graph.adj_list
    rev_adj_list = graph.rev_adj_list

    for _ in range(tries):
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

        graph_cpy = deepcopy(graph)

        graph_cpy.delete_node(random_operator)
        graph_cpy.add_edges(edges_added)

        # check if the graph is still valid
        if not check_graph_validity(graph_cpy)[0]:
            continue

        return graph_cpy, True

    return graph, False


def add_subgraph(graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0) -> tuple[graph_.Graph, bool]:
    random_num_inputs = random.randint(2, 5)

    node_specs = [*[{"name": "input"} for _ in range(random_num_inputs)], {"name": "add"}, {"name": "output"}]

    rev_adj_list: list[list[int]] = [
        *[[] for _ in range(random_num_inputs)],
        [i for i in range(random_num_inputs)],
        [random_num_inputs],
    ]

    input_node_order = list(range(random_num_inputs))

    subgraph = graph_.make_graph(
        node_specs=node_specs, rev_adj_list=rev_adj_list, input_node_order=input_node_order, is_subgraph=True
    )

    subgraph = mutate_graph(subgraph, mutation_hps, subgraph_depth=subgraph_depth + 1)

    graph_copy = deepcopy(graph)

    graph_copy.subgraphs.append(subgraph)

    return graph_copy, True


def remove_subgraph(
    graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0
) -> tuple[graph_.Graph, bool]:
    if len(graph.subgraphs) == 0:
        return graph, False

    graph_copy = deepcopy(graph)

    # only remove subgraphs that do not appear in the graph
    subgraphs_to_delete = graph_copy.subgraphs
    for node in graph_copy.id_to_node.values():
        if isinstance(node, nodes_.SubGraphNode):
            subgraphs_to_delete = [s for s in subgraphs_to_delete if s is not node.subgraph]

    if len(subgraphs_to_delete) == 0:
        return graph, False

    # choose the subgraph randomly
    subgraph_idx = random.randint(0, len(subgraphs_to_delete) - 1)
    subgraph_to_delete = subgraphs_to_delete[subgraph_idx]

    graph_copy.subgraphs = [s for s in graph_copy.subgraphs if s is not subgraph_to_delete]

    return graph_copy, True


def swap_incoming_edge_source(
    graph: graph_.Graph, mutation_hps: GraphMutHP, subgraph_depth: int = 0, tries: int = DEFAULT_NUM_TRIES
) -> tuple[graph_.Graph, bool]:
    for _ in range(tries):
        graph_copy = deepcopy(graph)

        opnodes = graph_copy.operator_nodes()
        if len(opnodes) == 0:
            return graph, False

        node_id = random.choice(opnodes)

        incoming_edges = graph_copy.rev_adj_list[node_id]

        if len(incoming_edges) == 0:
            continue

        old_source_to_delete = random.choice(incoming_edges)
        candidate_sources = set(graph_copy.id_to_node.keys()) - set(incoming_edges)
        if graph_copy.is_subgraph:
            candidate_sources -= set(graph_copy.output_nodes())

        new_source = random.choice(list(candidate_sources))

        graph_copy.delete_edge(old_source_to_delete, node_id)
        graph_copy.add_edge(new_source, node_id)

        if not check_graph_validity(graph_copy)[0]:
            continue

        return graph_copy, True

    return graph, False


def mutate_graph(graph: graph_.Graph, mutation_hyperparams: GraphMutHP, subgraph_depth: int = 0) -> graph_.Graph:
    mutated = graph

    max_num_mutations = mutation_hyperparams.max_num_mutations

    if graph.is_subgraph:
        mutation_probs = deepcopy(mutation_hyperparams.subgraph_mutation_probabilities)
    else:
        mutation_probs = deepcopy(mutation_hyperparams.mutation_probabilities)

    num_subgraphs = len(graph.subgraphs)
    if (
        subgraph_depth >= mutation_hyperparams.max_subgraph_depth
        or num_subgraphs >= mutation_hyperparams.max_num_subgraphs
    ):
        mutation_probs.pop("add_subgraph")

    names, probs = zip(*list(mutation_probs.items()))

    mutations_performed = 0
    if graph.is_subgraph:
        num_mutations = 1
    else:
        num_mutations = weighted_random(1, max_num_mutations)

    while mutations_performed < num_mutations:
        mutation_name = random.choices(names, weights=probs)[0]
        mutation_function = graph_mutation_functions[mutation_name]

        mutated, changed = mutation_function(mutated, mutation_hyperparams, subgraph_depth)
        valid, reason = check_graph_validity(mutated)
        if not valid:
            graph_.show_graph(mutated, show_node_ids=True)
            raise RuntimeError(f"Tried to mutate with {mutation_name}, but graph is invalid: {reason}")

        if changed:
            mutations_performed += 1
            # print(f"Mutated with {mutation_name}, {mutations_performed}/{num_mutations} mutations performed")

    return mutated


def mutate_subgraph(
    graph: graph_.Graph, mutation_hyperparameters: GraphMutHP, subgraph_depth: int = 0
) -> tuple[graph_.Graph, bool]:
    if len(graph.subgraphs) == 0:
        return graph, False

    subgraph_idx = random.randint(0, len(graph.subgraphs) - 1)

    subgraph = graph.subgraphs[subgraph_idx]
    subgraph_copy = deepcopy(subgraph)
    subgraph_copy = mutate_graph(subgraph_copy, mutation_hyperparameters, subgraph_depth=subgraph_depth + 1)

    graph_cpy = deepcopy(graph)

    graph_cpy.subgraphs[subgraph_idx] = subgraph_copy

    for node_id, node in graph.id_to_node.items():
        if isinstance(node, nodes_.SubGraphNode) and node.subgraph is subgraph:
            graph_cpy.id_to_node[node_id] = nodes_.SubGraphNode(subgraph_copy)

    return graph_cpy, True


graph_mutation_functions: dict[str, MutFunction] = {
    "expand_edge": expand_edge,
    "add_edge": add_edge,
    "add_parameter": add_parameter,
    "delete_operator": delete_operator,
    "delete_edge": delete_edge,
    "delete_parameter": delete_parameter,
    "mutate_subgraph": mutate_subgraph,
    "add_subgraph": add_subgraph,
    "remove_subgraph": remove_subgraph,
    "swap_incoming_edge_source": swap_incoming_edge_source,
}
