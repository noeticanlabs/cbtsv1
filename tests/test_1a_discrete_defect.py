"""
Test 1A: Discrete Defect (MMS Lie Detector)
"""

import numpy as np
import logging
from tests.gr_test_utils import rms, l2, estimate_order
from tests.gr_adapter import NRAdapter, create_gr_adapter

def gcat1_test1a_discrete_defect(
    adapter: NRAdapter,
    Ns=(16, 32, 64),
    L=16.0,
    CFL=0.01,
    dt_mode="dx2",     # "dx" or "dx2" (you used dx^2 to suppress time error)
    compare_keys=None, # None -> compare all keys returned
    use_rms=True,
    verbose=True
) -> dict:
    """
    Returns:
      - defects_no_sources[N]
      - defects_with_sources[N]
      - p_defect_with_sources
      - classification
    """
    defects0 = []
    defectsS = []
    hs = []

    for N in Ns:
        dx = L / N
        dt = 0.01  # Fixed dt for one step
        t0 = 0.0
        t1 = t0 + dt

        # Build fields and set to exact state at t0
        fields0 = adapter.make_fields(N, L)
        adapter.set_exact_solution(fields0, t0)

        # Reference exact arrays
        U_exact_t0 = adapter.exact_state_arrays(N, L, t0)
        U_exact_t1 = adapter.exact_state_arrays(N, L, t1)

        # Step WITHOUT sources
        f_in = adapter.clone_fields(fields0) if adapter.clone_fields else fields0
        f_nosrc = adapter.step(f_in, dt, sources=None)
        U_nosrc = adapter.get_state_arrays(f_nosrc)

        # Check initialization
        U_init = adapter.get_state_arrays(fields0)
        for k in U_init:
            err_init = U_init[k] - U_exact_t0[k]
            err_init_norm = np.linalg.norm(err_init) / np.sqrt(err_init.size)
            print(f"[Init check] N={N} {k} error: {err_init_norm:.2e}")

        # Step WITH MMS sources (time-dependent for RK stages)
        sources_func = lambda t: adapter.make_mms_sources(N, L, t)
        f_in2 = adapter.clone_fields(fields0) if adapter.clone_fields else fields0
        f_src = adapter.step(f_in2, dt, sources=sources_func)
        U_src = adapter.get_state_arrays(f_src)

        # Compare selected keys
        keys = compare_keys if compare_keys is not None else list(U_nosrc.keys())
        # Defect is: U_exact(t1) - U_num(after 1 step)
        d0_list = []
        dS_list = []
        for k in keys:
            a1 = U_exact_t1[k]
            b0 = U_nosrc[k]
            bS = U_src[k]
            d0 = rms(a1 - b0) if use_rms else l2(a1 - b0)
            dS = rms(a1 - bS) if use_rms else l2(a1 - bS)
            d0_list.append(d0)
            dS_list.append(dS)
            print(f"[Defect] N={N} {k} defect_no_src: {d0:.2e} defect_with_src: {dS:.2e}")

        defect0 = float(np.max(d0_list))  # max over variables = conservative
        defectS = float(np.max(dS_list))

        defects0.append(defect0)
        defectsS.append(defectS)
        hs.append(dx)

        if verbose:
            print(".3e")

    pS = estimate_order(defectsS, hs)  # should be ~ +2 for 2nd order in space, etc. (NOTE: p is exponent on h)
    p0 = estimate_order(defects0, hs)

    # Plateau detector (flat errors)
    def is_plateau(errs, tol=0.15):
        if len(errs) < 3:
            return False
        r1 = errs[0] / max(errs[1], 1e-300)
        r2 = errs[1] / max(errs[2], 1e-300)
        return (abs(r1 - 1.0) < tol) and (abs(r2 - 1.0) < tol)

    plateauS = is_plateau(defectsS)
    plateau0 = is_plateau(defects0)

    # Classification logic
    if plateauS and not plateau0:
        classification = "FAIL_MMS_DISCRETE_MISMATCH (sources not aligned to discrete scheme)"
    elif plateauS and plateau0:
        classification = "FAIL_DX_OR_STEP_PLUMBING (operator/step not refining with N, or comparing wrong vars)"
    elif (pS < 1.5):  # expecting >1.5 if you want at least ~2nd order
        classification = f"FAIL_LOW_ORDER (p_defect_with_sources={pS:.2f})"
    else:
        classification = f"PASS (p_defect_with_sources={pS:.2f})"

    return {
        "passed": (pS >= 1.5) and (not plateauS),
        "metrics": {
            "Ns": list(Ns),
            "dx": hs,
            "defect_no_sources": defects0,
            "defect_with_sources": defectsS,
            "p_defect_no_sources": p0,
            "p_defect_with_sources": pS,
            "plateau_no_sources": plateau0,
            "plateau_with_sources": plateauS,
        },
        "diagnosis": classification,
    }

class Test1A:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        adapter = create_gr_adapter()
        result = gcat1_test1a_discrete_defect(adapter)
        return result