"""
Test 1B: Operator-only convergence (dx detector)
"""

import numpy as np
import logging
from tests.gr_test_utils import rms, estimate_order
from tests.gr_adapter import NRAdapter, create_gr_adapter

def gcat1_test1b_operator_convergence(
    adapter: NRAdapter,
    Ns=(16, 32, 64),
    L=16.0,
    mode="1d",     # "1d" or "3d" analytic function
    expected_order=2.0,
    verbose=True
) -> dict:
    """
    Tests Dx/Dxx/Lap etc on analytic functions.
    """
    if adapter.Dx is None or adapter.Dxx is None or adapter.Lap is None:
        raise ValueError("Adapter must provide Dx, Dxx, Lap at minimum for Test1B.")

    errs = {"Dx": [], "Dxx": [], "Lap": []}
    hs = []

    for N in Ns:
        dx = L / N
        x = np.linspace(0.0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

        # Analytic field
        k = 2.0 * np.pi / L
        if mode == "1d":
            f = np.sin(k * X)  # varies in x only
            Dx_exact  = k * np.cos(k * X)
            Dxx_exact = -(k * k) * np.sin(k * X)
            Lap_exact = Dxx_exact
        else:
            f = np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
            Dx_exact  = k * np.cos(k * X) * np.sin(k * Y) * np.sin(k * Z)
            Dxx_exact = -(k * k) * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
            Lap_exact = -3.0 * (k * k) * f

        Dx_num  = adapter.Dx(f, dx)
        Dxx_num = adapter.Dxx(f, dx)
        Lap_num = adapter.Lap(f, dx)

        eDx  = rms(Dx_num - Dx_exact)
        eDxx = rms(Dxx_num - Dxx_exact)
        eLap = rms(Lap_num - Lap_exact)

        errs["Dx"].append(float(eDx))
        errs["Dxx"].append(float(eDxx))
        errs["Lap"].append(float(eLap))
        hs.append(dx)

        if verbose:
            print(f"[Test1B] N={N} dx={dx:.3e} eDx={eDx:.3e} eDxx={eDxx:.3e} eLap={eLap:.3e}")

    pDx  = estimate_order(errs["Dx"], hs)
    pDxx = estimate_order(errs["Dxx"], hs)
    pLap = estimate_order(errs["Lap"], hs)

    # Pass rule: each operator meets expected-ish order (loose)
    # Use 0.5 margin because stencils, filtering, etc. can reduce observed slope a bit.
    pass_ops = (pDx >= expected_order - 0.5) and (pDxx >= expected_order - 0.5) and (pLap >= expected_order - 0.5)

    diagnosis = f"{'PASS' if pass_ops else 'FAIL'} (pDx={pDx:.2f}, pDxx={pDxx:.2f}, pLap={pLap:.2f})"
    if not pass_ops:
        diagnosis += " -> likely dx misuse, wrong stencil scaling, or operator not refining."

    return {
        "passed": bool(pass_ops),
        "metrics": {
            "Ns": list(Ns),
            "dx": hs,
            "errors": errs,
            "pDx": pDx,
            "pDxx": pDxx,
            "pLap": pLap,
        },
        "diagnosis": diagnosis,
    }

class Test1B:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        adapter = create_gr_adapter()
        result = gcat1_test1b_operator_convergence(adapter)
        return result