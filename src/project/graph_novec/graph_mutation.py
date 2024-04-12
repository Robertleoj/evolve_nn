"""Mutation operators for graphs"""
from project.graph_novec.graph import Graph, show_graph
from project.graph_novec.nodes import DataNode, OperatorNode, operator_nodes, node_from_name, ParameterNode
from project.type_defs import EvolutionConfig
import random
import networkx as nx
from dataclasses import dataclass
from copy import deepcopy

def check_validity(graph: Graph):
    # 1: the graph must be weakly connected
    g_nx = graph.get_nx()
    
    assert nx.is_weakly_connected(g_nx), "Graph is not weakly connected"
    
    # 2: the graph must be acyclic
    assert nx.is_directed_acyclic_graph(g_nx), "Graph is not acyclic"
    
    adj_list = graph.adjacency_list()
    rev_adj_list = graph.adjacency_list(reverse=True)
    # 3: input nodes must have no incoming edges
    for node_id in graph.input_nodes():
        assert len(rev_adj_list[node_id]) == 0, f"Input node {node_id} has incoming edges"

    # 4: output nodes must have no outgoing edges
    for node_id in graph.output_nodes():
        assert len(adj_list[node_id]) == 0, f"Output node {node_id} has outgoing edges"
        
    # 5: output nodes must have exactly one incoming edge
    for node_id in graph.output_nodes():
        assert len(rev_adj_list[node_id]) == 1, f"Data node {node_id} does not have exactly one incoming edge"

    # 6: Operator nodes have at least one outgoing edge
    for node_id in graph.operator_nodes():
        assert len(adj_list[node_id]) > 0, f"Operator node {node_id} does not have outgoing edges"

    # 7: Operator nodes have number of input nodes that is within the bounds
    for node_id in graph.operator_nodes():
        lower_bound, upper_bound = graph.id_to_node[node_id].n_inputs
        n_inputs = len(rev_adj_list[node_id])
        assert n_inputs >= lower_bound, f"Operator node {node_id} has {n_inputs} inputs, expected at least {lower_bound}"
        if upper_bound is not None:
            assert n_inputs <= upper_bound, f"Operator node {node_id} has {n_inputs} inputs, expected at most {upper_bound}"
    
    

def available_opnodes(graph: Graph, no_single_params: bool = False) -> list[str]:
    opnodes = graph.operator_nodes()
    adj_list = graph.adjacency_list(reverse=True)
    forward_adj_list = graph.adjacency_list()
    
    available_nodes = []

    for node_id in opnodes:
        inputs = adj_list[node_id]
        if no_single_params:
            found_single_param = False
            for i in inputs:
                if isinstance(graph.id_to_node[i], ParameterNode):
                    if len(set(forward_adj_list[i])) == 1:
                        found_single_param = True
                        break
            if found_single_param:
                continue

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
    available_nodes = available_opnodes(graph, no_single_params=True)

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

    output_nodes = set(graph.output_nodes())

    node_1_candidates = [
        node_id for node_id in graph.node_ids
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

def delete_parameter(graph: Graph, tries=30) -> tuple[Graph, bool]:
    paramnodes = [
        node_id for node_id in graph.node_ids
        if isinstance(graph.id_to_node[node_id], ParameterNode)
    ]

    if len(paramnodes) == 0:
        return graph, False


    adj_list = graph.adjacency_list()
    rev_adj_list = graph.adjacency_list(reverse=True)
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

        graph_copy.reset_edge_list([
            edge for edge in graph_copy.edge_list
            if random_parameter not in edge
        ])

        graph_copy.remove_node(random_parameter)

        return graph_copy, True

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

        graph_cpy = deepcopy(graph)

        graph_cpy.reset_edge_list([
            edge for edge in graph_cpy.edge_list
            if random_operator not in edge
        ] + edges_added)

        graph_cpy.remove_node(random_operator)

        new_adj_list = graph_cpy.adjacency_list()
        all_ok = True
        for node_id in graph_cpy.node_ids:
            # we also want to make sure all parameters point to operators
            if isinstance(graph_cpy.id_to_node[node_id], ParameterNode):
                if len(new_adj_list[node_id]) == 0:
                    all_ok = False
                    break
                
                only_has_operator_nodes = True
                for to_node_id in new_adj_list[node_id]:
                    if not isinstance(graph_cpy.id_to_node[i], OperatorNode):
                        only_has_operator_nodes = False
                        break
                if not only_has_operator_nodes:
                    all_ok = False
                    break

            # all operator nodes must have outgoing edges
            if isinstance(graph_cpy.id_to_node[node_id], OperatorNode):
                if len(new_adj_list[node_id]) == 0:
                    all_ok = False
                    break

        if not all_ok:
            continue

        return graph_cpy, True
    
    return graph, False

mutation_functions = {
    'expand_edge': expand_edge,
    'add_edge': add_edge,
    'add_parameter': add_parameter,
    'delete_operator': delete_operator,
    'delete_edge': delete_edge,
    'delete_parameter': delete_parameter
}

@dataclass
class GraphMutHP:
    max_num_mutations: int
    mutation_probabilities: dict[str, float]

def random_graph_mut_hps(evolution_config: EvolutionConfig) -> GraphMutHP:
    probs = {}
    for k in mutation_functions.keys():
        probs[k] = random.uniform(0, 1)

    if evolution_config.mutate_num_mutations:
        max_num_mutations=random.randint(1, 10)
    else:
        assert evolution_config.max_num_mutations is not None
        max_num_mutations = evolution_config.max_num_mutations

    return GraphMutHP(
        mutation_probabilities=probs,
        max_num_mutations=max_num_mutations
    )

def mutate_graph(graph: Graph, mutation_hyperparams: GraphMutHP) -> Graph:
    mutated = graph

    max_num_mutations = mutation_hyperparams.max_num_mutations
    names, probs = zip(*list(mutation_hyperparams.mutation_probabilities.items()))

    mutations_performed = 0
    num_mutations = random.randint(1, max_num_mutations)
    while mutations_performed < num_mutations:
        mutation_name = random.choices(names, weights=probs)[0]
        mutation_function = mutation_functions[mutation_name]
        mutated, changed = mutation_function(mutated)

        try:
            check_validity(mutated)
        except Exception as e:
            print("Tried to mutate with {} but failed".format(mutation_function))
            print(e)
            show_graph(mutated)
            raise e

        if changed:
            mutations_performed += 1

    return mutated

def mutate_graph_hps(
    hp: GraphMutHP, 
    evolution_config: EvolutionConfig
):
    new_hp = deepcopy(hp)

    if evolution_config.mutate_num_mutations:
        new_hp.max_num_mutations = max(hp.max_num_mutations + random.choice([-1, 0, 1]), 1)
    else:
        new_hp.max_num_mutations = hp.max_num_mutations

    new_probs = {}
    for k, v in hp.mutation_probabilities.items():
        new_probs[k] = v * random.uniform(0.8, 1.2)

    new_hp.mutation_probabilities = new_probs

    return new_hp
    
def recombine_graph_hps(hp1: GraphMutHP, hp2: GraphMutHP, evolution_config: EvolutionConfig) -> GraphMutHP:
    new_hp = deepcopy(hp1)

    if evolution_config.mutate_num_mutations:
        new_hp.max_num_mutations = random.choice([hp1.max_num_mutations, hp2.max_num_mutations])
    else:
        new_hp.max_num_mutations = hp1.max_num_mutations

    new_probs = {}
    for k in hp1.mutation_probabilities.keys():
        new_probs[k] = random.choice([hp1.mutation_probabilities[k], hp2.mutation_probabilities[k]])

    new_hp.mutation_probabilities = new_probs

    return new_hp

def recombine_graphs(graph1: Graph, graph2: Graph, hp: GraphMutHP) -> Graph:
    return graph1