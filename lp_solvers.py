import networkx as nx
from ortools.linear_solver import pywraplp
from pulp import LpProblem, LpMaximize, LpVariable, LpStatusOptimal, value, lpSum
import pulp
import time

or_tools_solvers = ['GLOP', 'PDLP', 'CLP']

pulp_solvers_local = {
    'PULP_CBC_CMD': pulp.PULP_CBC_CMD(msg=0),
    'PULP_GLPK_PRIMAL': pulp.GLPK_CMD(msg=0, options=['primalSimplex']),
    'PULP_GLPK_DUAL': pulp.GLPK_CMD(msg=0, options=['dualSimplex']),
    'PULP_GLPK_BARRIER': pulp.GLPK_CMD(msg=0, options=['barrier']),
    'PULP_COIN_CMD': pulp.COIN_CMD(msg=0)
}

pulp_solvers_colab = {
    'PULP_CBC_CMD': pulp.PULP_CBC_CMD(msg=0)
}


def max_flow_with_ortools(G, s, t, method='GLOP', debug=False):
    if method not in ['GLOP', 'PDLP', 'CLP']:
        raise ValueError("Invalid solver method. Please choose from 'GLOP', 'PDLP', or 'CLP'.")

    # Create the LP solver
    solver = pywraplp.Solver.CreateSolver(method)

    # Create variables for flows on edges
    flow = {}
    for u, v, data in G.edges(data=True):
        capacity = data.get('capacity', 1)  # Default capacity is 1 if not provided
        var_name = f'f_{u}_{v}'
        flow[(u, v)] = solver.NumVar(0, capacity, var_name)

    # Flow conservation constraints for all nodes except s and t
    for node in G.nodes():
        if node == s or node == t:
            continue

        # Calculate inflow and outflow manually
        inflow = solver.Sum([flow[(u, node)] for u, v in G.edges() if v == node and (u, v) in flow])
        outflow = solver.Sum([flow[(node, v)] for u, v in G.edges() if u == node and (u, v) in flow])
        solver.Add(inflow == outflow)

    # Objective: Maximize total flow into sink t
    total_inflow_t = solver.Sum([flow[(u, t)] for u, v in G.edges() if v == t and (u, v) in flow])
    total_outflow_t = solver.Sum([flow[(t, v)] for u, v in G.edges() if u == t and (u, v) in flow])
    net_flow_t = total_inflow_t - total_outflow_t
    solver.Maximize(net_flow_t)

    if debug:
        print(f"Using the {method} method between {s} and {t}")

    # Solve the LP problem
    start = time.perf_counter()
    status = solver.Solve()
    end = time.perf_counter()

    solve_time = end - start

    if status == pywraplp.Solver.OPTIMAL:
        max_flow_value = net_flow_t.solution_value()
        if debug:
            print(f"Maximum flow from {s} to {t}: {max_flow_value}")
            # Optionally, print the flow values on the edges
            for (u, v), var in flow.items():
                flow_value = var.solution_value()
                if flow_value > 0:
                    print(f"Flow from {u} to {v}: {flow_value}")
    else:
        max_flow_value = None
        if debug:
            print("The solver did not find an optimal solution.")
            print(f"LP Solve Time: {solve_time:.6f} seconds")

    result = {
        'run_time': solve_time,
        'flow_value': max_flow_value,
        'status': status == pywraplp.Solver.OPTIMAL,
    }

    return result


def max_flow_with_pulp(G, s, t, method='PULP_CBC_CMD', debug=False):
    # Get list of node indices
    nodes = list(G.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must have at least two nodes.")

    # Create the LP problem
    prob = LpProblem("MaxFlow", LpMaximize)

    # Create variables for flows on edges
    flow = {}
    # Build mapping from nodes to incoming and outgoing edges
    in_edges = {node: [] for node in nodes}
    out_edges = {node: [] for node in nodes}

    # Iterate over edges to create flow variables
    for u, v, data in G.edges(data=True):
        capacity = data.get('capacity')  # Default capacity is 1 if not provided
        var_name = f'f_{u}_{v}'
        flow[(u, v)] = LpVariable(var_name, lowBound=0, upBound=capacity)
        out_edges[u].append((u, v))
        in_edges[v].append((u, v))

    # Flow conservation constraints for all nodes except s and t
    for node in nodes:
        if node == s or node == t:
            continue
        inflow = lpSum([flow[(u, node)] for (u, node) in in_edges[node]])
        outflow = lpSum([flow[(node, v)] for (node, v) in out_edges[node]])
        prob += (inflow == outflow), f"FlowConservation_{node}"

    # Objective: Maximize total flow into sink t
    total_inflow_t = lpSum([flow[(u, t)] for (u, t) in in_edges[t]])
    total_outflow_t = lpSum([flow[(t, v)] for (t, v) in out_edges[t]])
    net_flow_t = total_inflow_t - total_outflow_t
    prob += net_flow_t, "MaximizeNetFlowIntoSink"

    solver_type = pulp_solvers_local[method]

    if debug:
        print(f"Using solver: {method} between nodes {s} and {t}")

    # Time the LP solve
    start_time = time.perf_counter()
    status = prob.solve(solver_type)
    end_time = time.perf_counter()

    solve_time = end_time - start_time  # Elapsed time in seconds

    if status == LpStatusOptimal:
        max_flow_value = value(net_flow_t)
        if debug:
            print(f"Maximum flow from {G[s]} to {G[t]}: {max_flow_value}")
            print(f"LP Solve Time: {solve_time:.6f} seconds")  # Print the solve time
            # Optionally, print the flow values on the edges
            for (u, v), var in flow.items():
                flow_value = var.varValue
                if flow_value > 0:
                    print(f"Flow from {G[u]} to {G[v]}: {flow_value}")
    else:
        max_flow_value = None
        if debug:
            print("The solver did not find an optimal solution.")
            print(f"LP Solve Time: {solve_time:.6f} seconds")  # Even if not optimal, print the solve time

    result = {
        'run_time': solve_time,
        'flow_value': max_flow_value,
        'status': status == LpStatusOptimal
    }

    return result


# Single OR Tools wrapper functions to call individual methods
def max_flow_ortools_GLOP(G, s, t):
    return max_flow_with_ortools(G, s, t, method='GLOP')


def max_flow_ortools_PDLP(G, s, t):
    return max_flow_with_ortools(G, s, t, method='PDLP')


def max_flow_ortools_CLP(G, s, t):
    return max_flow_with_ortools(G, s, t, method='CLP')


# Single Pulp wrapper functions to call individual methods
def max_flow_pulp_CBC(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_CBC_CMD')


def max_flow_pulp_GLPK_primal(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_GLPK_PRIMAL')


def max_flow_pulp_GLPK_dual(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_GLPK_DUAL')


def max_flow_pulp_GLPK_barrier(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_GLPK_BARRIER')


def max_flow_pulp_COIN(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_COIN_CMD')


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_edge('A', 'B', capacity=3)
    G.add_edge('A', 'C', capacity=2)
    G.add_edge('B', 'C', capacity=1)
    G.add_edge('B', 'D', capacity=3)
    G.add_edge('C', 'D', capacity=2)
    G.add_edge('C', 'E', capacity=3)
    G.add_edge('D', 'E', capacity=2)

    print('-' * 50)
    print("Solving example problem with ortools")
    response = max_flow_with_ortools(G, 'B', 'E', method='CLP', debug=True)
    print(response)
    print('-' * 50)

    print('-' * 50)
    print("Solving example problem with pulp")
    response = max_flow_with_pulp(G, 'B', 'E', debug=True)
    print(response)
    print('-' * 50)
