import numpy as np
from tests.test_gcat1_calibration_suite import create_gr_adapter
from src.core.gr_solver import GRSolver

# Fixed N=16, vary dt
N = 16
dx = 1.0
L = N * dx
adapter = create_gr_adapter()

# Exact at T=1.0
T = 1.0
U_exact_final = adapter.exact_state_arrays(N, L, T)

# dt values: start large, halve each time
dts = [0.01, 0.005, 0.0025, 0.00125]

errors = []
for dt in dts:
    fields = adapter.make_fields(N, L)
    adapter.set_exact_solution(fields, 0.0)
    
    # Set sources
    sources = adapter.make_mms_sources(N, L, 0.0)
    fields.stepper.sources_func = lambda t: adapter.make_mms_sources(N, L, t)
    
    # Force dt
    fields.scheduler.fixed_dt = dt
    fields.orchestrator.t = 0.0
    fields.orchestrator.step = 0
    
    while fields.orchestrator.t < T:
        dt_max = T - fields.orchestrator.t
        dt_actual, _, _ = fields.orchestrator.run_step(dt_max)
    
    U_num = adapter.get_state_arrays(fields)
    
    # Compute error
    err_list = []
    for k in U_exact_final.keys():
        if k in U_num:
            err = np.sqrt(np.mean((U_exact_final[k] - U_num[k])**2))
            err_list.append(err)
    error = max(err_list)  # max over fields
    errors.append(error)
    print(f'dt={dt:.6f}, error={error:.6e}')

# Compute p_time
import math
p_times = []
for i in range(1, len(errors)):
    if errors[i-1] > 0:
        p = math.log2(errors[i-1] / errors[i])
        p_times.append(p)
avg_p = np.mean(p_times) if p_times else 0
print(f'Average p_time: {avg_p:.2f}')