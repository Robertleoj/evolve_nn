import networkx as nx


def topsort_adj_list(adj_list: list[list[int]]) -> list[int]:
    """Topologically sort a graph given its adjacency list.

    Args:
        adj_list: Adjacency list of the graph.

    Returns:
        List of node indices in topological order.
    """
    return list(nx.topological_sort(nx.DiGraph(adj_list)))


def topsort_edge_list(num_nodes: int, edge_list: list[tuple[int, int]]) -> list[int]:
    """Topologically sort a graph given its edge list.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_list: List of edges in the graph.

    Returns:
        List of node indices in topological order.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_list)
    return list(nx.topological_sort(graph))


def reverse_adjacency_list(adj_list: list[list[int]]) -> list[list[int]]:
    """Reverse the adjacency list of a graph.

    Args:
        adj_list: Adjacency list of the graph.

    Returns:
        Reversed adjacency list.
    """
    n = len(adj_list)
    rev_adj_list: list[list[int]] = [[] for _ in range(n)]
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            rev_adj_list[j].append(i)
    return rev_adj_list


def are_all_reachable(G: nx.DiGraph, A: set[str], B: set[str]) -> bool:
    reachable_from_A = set()
    for start_node in A:
        reachable_from_A.update(nx.descendants(G, start_node))

    return B.issubset(reachable_from_A)
