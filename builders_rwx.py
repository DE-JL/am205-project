import random
import rustworkx as rx
from itertools import combinations
import heapq
import numpy as np


def gen_complete(n):
    # Create a directed graph
    G = rx.PyDiGraph()
    G.add_nodes_from(range(n))

    # Add edges with random capacities
    edge_data = []
    for u, v in combinations(range(n), 2):
        capacity = random.uniform(0, 1)
        edge_data.append((u, v, {"capacity": capacity}))
        edge_data.append((v, u, {"capacity": capacity}))

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

    G = rx.PyDiGraph()
    G.add_nodes_from(range(n))

    # For each undirected edge (u, v), add both (u, v) and (v, u) with the same capacity
    for (u, v) in edges:
        capacity = random.random()
        G.add_edge(u, v, {"capacity": capacity})
        G.add_edge(v, u, {"capacity": capacity})

    return G


def gen_erdos_renyi(n, p):
    # Generate a directed Erdős–Rényi graph using rustworkx
    G = rx.directed_gnp_random_graph(n, p)

    # Assign random capacities to each edge
    for u, v in G.edge_list():
        capacity = random.random()  # Generate a random capacity
        G.update_edge(u, v, {"capacity": capacity})

    return G


def gen_barabasi_albert(n, m):
    # Generate an undirected Barabasi-Albert graph using rustworkx
    G = rx.barabasi_albert_graph(n, m)

    # Calculate and assign capacities for each edge
    for u, v in G.edge_list():
        # Compute the capacity as the sum of the degrees of u and v
        degree_u = G.degree(u)  # Degree of u
        degree_v = G.degree(v)  # Degree of v
        capacity = degree_u + degree_v

        # Update the edge with the capacity
        G.update_edge(u, v, {"capacity": capacity})

    return G


def gen_sparse_ill_conditioned(n, edge_factor=1.5):
    # Create an undirected graph
    G = rx.PyGraph()
    G.add_nodes_from([None] * n)  # Add n nodes with no data

    # Generate a random tree to ensure connectivity
    nodes = list(range(n))
    random.shuffle(nodes)
    edges = []
    for i in range(1, n):
        u = nodes[i - 1]
        v = nodes[i]
        edges.append((u, v, None))

    # Add edges of the tree to the graph
    G.add_edges_from(edges)

    existing_edges = set((min(u, v), max(u, v)) for u, v, _ in edges)

    # Calculate desired number of edges
    desired_num_edges = int(edge_factor * n)

    # Add random edges until we reach the desired number of edges
    while len(edges) < desired_num_edges:
        u, v = random.sample(range(n), 2)
        edge = (min(u, v), max(u, v))
        if edge not in existing_edges:
            G.add_edge(u, v, None)
            edges.append((u, v, None))
            existing_edges.add(edge)

    # Create a directed graph and add both directions for each edge
    DG = rx.PyDiGraph()
    DG.add_nodes_from([None] * n)  # Add n nodes with no data

    # For capacities, half small, half large
    undirected_edges = [(u, v) for u, v, _ in edges]
    num_undirected_edges = len(undirected_edges)
    small_capacity = 1e-15
    large_capacity = 1e15

    # Shuffle edges to randomize capacity assignment
    random.shuffle(undirected_edges)

    # Assign capacities and add edges to the directed graph
    for i, (u, v) in enumerate(undirected_edges):
        capacity = small_capacity if i < num_undirected_edges // 2 else large_capacity
        DG.add_edge(u, v, {'capacity': capacity})
        DG.add_edge(v, u, {'capacity': capacity})

    return DG


def gen_graph(params, n):
    if params['type'] == 'complete':
        return gen_complete(n)
    if params['type'] == 'tree':
        return gen_trees(n)
    if params['type'] == 'erdos-renyi':
        return gen_erdos_renyi(n, params['p'])
    if params['type'] == 'barabasi-albert':
        return gen_barabasi_albert(n, params['m'])
    if params['type'] == 'sparse-ill-conditioned':
        return gen_sparse_ill_conditioned(n)
    return None


def sample_node_pairs(G, S):
    pairs = []
    for _ in range(S):
        u, v = np.random.choice(G.nodes(), size=2, replace=False)
        pairs.append((u, v))
    return pairs


def main():
    n = 500  # Number of nodes for all graphs
    examples = [
        {'type': 'complete'},
        {'type': 'tree'},
        {'type': 'erdos-renyi', 'p': 0.3},
        {'type': 'barabasi-albert', 'm': 2},
        {'type': 'sparse-ill-conditioned'},
    ]

    for params in examples:
        graph = gen_graph(params, n)

        # Print some basic properties of the graph
        print(f"Generated {params['type']} graph with {n} nodes:")
        print(f"  Number of edges: {len(graph.edge_list())}")
        print(f"  Sample edges with capacities:")

        for (u, v, data) in list(graph.edge_index_map().values())[:min(len(graph.edge_list()), 5)]:
            print(f"({u}, {v}) with capacity {data['capacity']}")
        print()


if __name__ == "__main__":
    main()
