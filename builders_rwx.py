import random
import rustworkx as rx
from itertools import combinations
import heapq
import numpy as np


""" COMPLETE """
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


""" TREES """
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


""" ERDOS-RENYI """
def gen_erdos_renyi(n, p):
    # Generate a directed Erdős–Rényi graph using rustworkx
    G = rx.directed_gnp_random_graph(n, p)

    # Assign random capacities to each edge
    for u, v in G.edge_list():
        capacity = random.random()  # Generate a random capacity
        G.update_edge(u, v, {"capacity": capacity})

    return G


""" Barabasi-Albert """
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


""" Generic """
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


def main():
    n = 10  # Number of nodes for all graphs
    examples = [
        {'type': 'complete'},
        {'type': 'tree'},
        {'type': 'erdos-renyi', 'p': 0.3},
        {'type': 'barabasi-albert', 'm': 2},
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
