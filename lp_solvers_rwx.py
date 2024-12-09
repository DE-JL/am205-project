import rustworkx as rx
from ortools.linear_solver import pywraplp
from pulp import LpProblem, LpMaximize, LpVariable, LpStatusOptimal, value, lpSum
import pulp
import time

or_tools_solvers = ['GLOP', 'PDLP', 'CLP']

pulp_solvers_local = {
    'PULP_CBC_CMD': pulp.PULP_CBC_CMD(msg=0),
    'PULP_GLPK': pulp.GLPK_CMD(msg=0),
    'PULP_COIN_CMD_PRIMAL': pulp.COIN_CMD(msg=0, options=['primalSimplex']),
    'PULP_COIN_CMD_DUAL': pulp.COIN_CMD(msg=0, options=['dualSimplex']),
    'PULP_COIN_CMD_BARRIER': pulp.COIN_CMD(msg=0, options=['barrier'])
}

def max_flow_with_ortools(G, s, t, method='GLOP', debug=False):
    
    if method not in ['GLOP', 'PDLP', 'CLP']:
        raise ValueError("Invalid solver method. Please choose from 'GLOP', 'PDLP', or 'CLP'.")

    # Create the LP solver
    solver = pywraplp.Solver.CreateSolver(method)

    # Create variables for flows on edges
    flow = {}
    for _, (u, v, data) in G.edge_index_map().items():
        capacity = data.get('capacity', 1)  # Default capacity is 1 if not provided
        var_name = f'f_{u}_{v}'
        flow[(u, v)] = solver.NumVar(0, capacity, var_name)

    # Flow conservation constraints for all nodes except s and t
    for node in range(len(G)):
        if node == s or node == t:
            continue
        inflow = solver.Sum([flow[(u, node)] for u, _, _ in G.in_edges(node) if (u, node) in flow])
        outflow = solver.Sum([flow[(node, v)] for _, v, _ in G.out_edges(node) if (node, v) in flow])
        solver.Add(inflow == outflow)

    # Objective: Maximize total flow into sink t
    total_inflow_t = solver.Sum([flow[(u, t)] for u, _, _ in G.in_edges(t) if (u, t) in flow])
    total_outflow_t = solver.Sum([flow[(t, v)] for _, v, _ in G.out_edges(t) if (t, v) in flow])
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

    # Create the LP problem
    prob = LpProblem("MaxFlow", LpMaximize)

    # Create variables for flows on edges
    flow = {}
    in_edges = {node: [] for node in range(len(G))}
    out_edges = {node: [] for node in range(len(G))}

    # Iterate over edges to create flow variables
    for _, (u, v, data) in G.edge_index_map().items():
        capacity = data.get('capacity', 1)  # Default capacity is 1 if not provided
        var_name = f'f_{u}_{v}'
        flow[(u, v)] = LpVariable(var_name, lowBound=0, upBound=capacity)
        out_edges[u].append((u, v))
        in_edges[v].append((u, v))

    # Flow conservation constraints for all nodes except s and t
    for node in range(len(G)):
        if node == s or node == t:
            continue
        inflow = lpSum([flow[(u, node)] for u, node in in_edges[node]])
        outflow = lpSum([flow[(node, v)] for node, v in out_edges[node]])
        prob += (inflow == outflow), f"FlowConservation_{node}"

    # Objective: Maximize total flow into sink t
    total_inflow_t = lpSum([flow[(u, t)] for u, t in in_edges[t]])
    total_outflow_t = lpSum([flow[(t, v)] for t, v in out_edges[t]])
    net_flow_t = total_inflow_t - total_outflow_t
    prob += net_flow_t, "MaximizeNetFlowIntoSink"

    solver_type = pulp_solvers_local[method]
    
    if debug:
        print(f"Using solver: {method} between nodes {s} and {t}")

    # Time the LP solve
    start_time = time.perf_counter()
    status = prob.solve(solver_type)
    end_time = time.perf_counter()

    solve_time = end_time - start_time

    if status == LpStatusOptimal:
        max_flow_value = value(net_flow_t)
        if debug:
            print(f"Maximum flow from {s} to {t}: {max_flow_value}")
            print(f"LP Solve Time: {solve_time:.6f} seconds")
            for (u, v), var in flow.items():
                flow_value = var.varValue
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
        'status': status == 1
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


def max_flow_pulp_COIN_primal(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_COIN_CMD_PRIMAL')


def max_flow_pulp_COIN_dual(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_COIN_CMD_DUAL')


def max_flow_pulp_COIN_barrier(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_COIN_CMD_BARRIER')


def max_flow_pulp_GLPK(G, s, t):
    return max_flow_with_pulp(G, s, t, method='PULP_GLPK')

if __name__ == "__main__":

    # Create a directed graph
    G = rx.PyDiGraph()

    # Add nodes
    nodes = {}
    for node in ['A', 'B', 'C', 'D', 'E']:
        nodes[node] = G.add_node(node)

    # Add edges with capacities
    G.add_edge(nodes['A'], nodes['B'], {"capacity": 3})
    G.add_edge(nodes['A'], nodes['C'], {"capacity": 2})
    G.add_edge(nodes['B'], nodes['C'], {"capacity": 1})
    G.add_edge(nodes['B'], nodes['D'], {"capacity": 3})
    G.add_edge(nodes['C'], nodes['D'], {"capacity": 2})
    G.add_edge(nodes['C'], nodes['E'], {"capacity": 3})
    G.add_edge(nodes['D'], nodes['E'], {"capacity": 2})

    # Solve the problem using OR-Tools
    print('-' * 50)
    print("Solving example problem with ortools")
    response = max_flow_with_ortools(G, nodes['B'], nodes['E'], method='CLP', debug=True)
    print(response)
    print('-' * 50)

    # Solve the problem using PuLP
    print('-' * 50)
    print("Solving example problem with pulp")
    response = max_flow_pulp_GLPK(G, nodes['B'], nodes['E'])
    print(response)
    print('-' * 50)
