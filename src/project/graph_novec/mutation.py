"""Mutation operators for graphs"""
from project.graph_novec.graph import Graph
from project.graph_novec.nodes import DataNode, OperatorNode, operator_nodes, node_from_name, ParameterNode
import random
import networkx as nx
from copy import deepcopy

def available_opnodes(graph: Graph) -> list[str]:
    opnodes = graph.operator_nodes()
    adj_list = graph.adjacency_list(reverse=True)
    
    available_nodes = []

    for node_id in opnodes:
        lower_bound, upper_bound = graph.id_to_node[node_id].n_inputs

        if upper_bound is None:
            # no upper bound
            available_nodes.append(node_id)
            continue

        n_inputs = len(adj_list[node_id])

        if n_inputs < upper_bound:
            available_nodes.append(node_id)

    return available_nodes

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

def add_parameter(graph: Graph) -> tuple[Graph, bool]:
    available_nodes = available_opnodes(graph)

    if len(available_nodes) == 0:
        return graph, False

    graph = deepcopy(graph)

    # create a new node
    parameter_node = ParameterNode()

    node_id = graph.add_node(parameter_node)

    random_node = random.choice(available_nodes)

    new_edge = (node_id, random_node)

    graph.edge_list.append(new_edge)

    return graph, True


def add_edge(graph: Graph, tries: int = 30) -> tuple[Graph, bool]:
    g_nx = graph.get_nx()

    output_nodes = set(graph.output_nodes())

    node_1_candidates = [
        node_id for node_id in graph.node_ids
        if node_id not in output_nodes
    ]

    node_2_candidates = available_opnodes(graph)

    for _ in range(tries):
        # get a random edge
        node1 = random.choice(node_1_candidates)
        node2 = random.choice(node_2_candidates)

        if nx.has_path(g_nx, node2, node1):
            continue

        graph = deepcopy(graph)
        graph.edge_list.append((node1, node2))

        return graph, True

    return graph, False

def delete_edge(graph: Graph, tries=100) -> tuple[Graph, bool]:
    g_nx = graph.get_nx()

    adj_list = graph.adjacency_list()
    rev_adj_list = graph.adjacency_list(reverse=True)

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
        graph.edge_list.remove((node1_id, node2_id))
        return graph, True
        
    return graph, False


def delete_operator(graph: Graph, tries=30) -> tuple[Graph, bool]:
    opnodes = graph.operator_nodes()

    if len(opnodes) == 0:
        return graph, False

    
    adj_list = graph.adjacency_list()
    rev_adj_list = graph.adjacency_list(reverse=True)

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
            for o in outgoing_nodes[len(incoming_nodes):]:
                edges_added.append((random.choice(incoming_nodes), o))
        
        for edge in edges_added:
            g_nx.add_edge(*edge)
            
        # delete the operator
        g_nx.remove_node(random_operator)

        if not nx.is_weakly_connected(g_nx):
            continue
        
        # graph is weakly connected - we can go ahead and remove the node

        graph = deepcopy(graph)

        graph.reset_edge_list([
            edge for edge in graph.edge_list
            if random_operator not in edge
        ] + edges_added)

        graph.remove_node(random_operator)

        return graph, True
    
    return graph, False

    

