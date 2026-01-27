import time
import json
import tracemalloc
import numpy as np
from src.core.gr_solver import GRSolver

def W0(mode):
    """Small grid GR solver workload."""
    aeonic = (mode == 'aeonic')
    Nx, Ny, Nz = 8, 8, 8
    T_max = 0.1
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.orchestrator.aeonic_mode = aeonic
    solver.init_minkowski()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        solver.run(T_max=T_max)
        correct = solver.t >= T_max - 1e-6
    except Exception as e:
        correct = False
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = end - start
    return {'time': elapsed, 'peak_alloc': peak, 'correct': correct}, (Nx, T_max)

def W1(mode):
    """Medium grid GR solver workload."""
    aeonic = (mode == 'aeonic')
    Nx, Ny, Nz = 16, 16, 16
    T_max = 0.2
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.orchestrator.aeonic_mode = aeonic
    solver.init_minkowski()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        solver.run(T_max=T_max)
        correct = solver.t >= T_max - 1e-6
    except Exception as e:
        correct = False
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = end - start
    return {'time': elapsed, 'peak_alloc': peak, 'correct': correct}, (Nx, T_max)

def W2(mode):
    """Large grid GR solver workload."""
    aeonic = (mode == 'aeonic')
    Nx, Ny, Nz = 32, 32, 32
    T_max = 0.5
    solver = GRSolver(Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0)
    solver.orchestrator.aeonic_mode = aeonic
    solver.init_minkowski()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        solver.run(T_max=T_max)
        correct = solver.t >= T_max - 1e-6
    except Exception as e:
        correct = False
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = end - start
    return {'time': elapsed, 'peak_alloc': peak, 'correct': correct}, (Nx, T_max)

def run_workload(workload, mode):
    """Run a workload in given mode, measure time and allocation, check correctness."""
    func = globals()[workload]
    result, param = func(mode)
    data = {
        'workload': workload,
        'mode': mode,
        'param': param,
        **result
    }
    return data

def output_jsonl(data, filename='output.jsonl'):
    """Append data to JSONL file."""
    with open(filename, 'a') as f:
        f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    for workload in ['W0', 'W1', 'W2']:
        for mode in ['aeonic', 'baseline']:
            data = run_workload(workload, mode)
            output_jsonl(data)