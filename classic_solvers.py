import time

from networkx.algorithms.flow import dinitz
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import preflow_push


def max_flow_dinitz(G, s, t):
    start_time = time.perf_counter()
    R = dinitz(G, s, t)
    end_time = time.perf_counter()

    run_time = end_time - start_time
    return {
        'status': True,
        'flow_value': R.graph['flow_value'],
        'run_time': run_time,
    }


def max_flow_edmonds_karp(G, s, t):
    start_time = time.perf_counter()
    R = edmonds_karp(G, s, t, capacity='capacity')
    end_time = time.perf_counter()

    run_time = end_time - start_time
    return {
        'status': True,
        'flow_value': R.graph['flow_value'],
        'run_time': run_time,
    }


def max_flow_preflow_push(G, s, t):
    start_time = time.perf_counter()
    R = preflow_push(G, s, t, capacity='capacity')
    end_time = time.perf_counter()

    run_time = end_time - start_time
    return {
        'status': True,
        'flow_value': R.graph['flow_value'],
        'run_time': run_time,
    }
