from itertools import combinations
import heapq
import numpy as np
import networkx as nx
import random


def gen_complete(n):
    # Create digraph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Random capacity for all nC2 edges
    edge_data = []
    for u, v in combinations(range(n), 2):
        capacity = random.uniform(0, 1)
        edge_data.append((u, v, {'capacity': capacity}))
        edge_data.append((v, u, {'capacity': capacity}))

    # Add edges
    G.add_edges_from(edge_data)

    return G


def gen_trees(n):
    def prufer_to_tree(prufer):
        degree = [1] * n
        edges = []

        # Increase degree for each node in prufer
        for x in prufer:
            degree[x] += 1

        # Leaves are nodes with degree=1
        leaves = [i for i, d in enumerate(degree) if d == 1]
        heapq.heapify(leaves)

        # Decode prufer: repeatedly connect smallest leaf to next code element
        for x in prufer:
            leaf = heapq.heappop(leaves)
            edges.append((leaf, x))
            degree[x] -= 1
            if degree[x] == 1:
                heapq.heappush(leaves, x)

        # Connect the last two leaves
        edges.append((heapq.heappop(leaves), heapq.heappop(leaves)))

        return edges

    # Generate a random prufer code and build the graph
    prufer = [random.randrange(n) for _ in range(n - 2)]
    edges = prufer_to_tree(prufer)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # For each undirected edge (u, v), add both (u, v) and (v, u) with the same capacity
    for (u, v) in edges:
        capacity = random.random()
        G.add_edge(u, v, capacity=capacity)
        G.add_edge(v, u, capacity=capacity)

    return G


def gen_erdos_renyi(n, p):
    # Generate a directed Erdős-Rényi graph G(n, p)
    G = nx.gnp_random_graph(n, p, directed=True)

    # Assign random capacities and add reverse edges with the same capacity
    for u, v in list(G.edges()):
        capacity = random.random()
        G[u][v]["capacity"] = capacity
        G.add_edge(v, u, capacity=capacity)
    
    # Guarantee that there is at least one edge
    G.add_edge(0, 1, capacity=random.random())

    return G


def gen_barabasi_albert(n, m):
    G = nx.barabasi_albert_graph(n, m)

    # Assign capacities based on degrees
    degrees = dict(G.degree())
    for u, v in G.edges():
        G[u][v]['capacity'] = degrees[u] + degrees[v]

    return G


def gen_graph(params, n):
    if params['type'] == 'complete':
        return gen_complete(n)
    if params['type'] == 'tree':
        return gen_trees(n)
    if params['type'] == 'erdos-renyi':
        return gen_erdos_renyi(n, params['p'])
    if params['type'] == 'barabasi-albert':
        return gen_barabasi_albert(n, params['m'])
    return None


def sample_node_pairs(G, S):
    pairs = []
    for _ in range(S):
        u, v = np.random.choice(G.nodes(), size=2, replace=False)
        pairs.append((u, v))
    return pairs
