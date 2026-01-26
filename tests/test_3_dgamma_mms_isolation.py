#!/usr/bin/env python3
"""
test_3_dgamma_mms_isolation.py
------------------------------
WP1 / GR calibration: isolate dgamma computation (derivative kernel) vs analytic truth.

Goal:
- Measure convergence of ∂_x gamma_ij, ∂_y gamma_ij, ∂_z gamma_ij for a smooth MMS metric.
- If dgamma does NOT converge (p_obs ~ 0), the GR operator pipeline cannot be trusted.

This test uses a *periodic* MMS by construction:
  omega = 2π * m / L
  s(x,y,z) = sin(omega x) sin(omega y) sin(omega z)
  gamma_ii = 1 + eps*s, off-diagonals = 0

Analytic derivatives:
  ∂x gamma_ii = eps * omega cos(omega x) sin(omega y) sin(omega z)
  (and cyclic)

Numerical derivatives:
- Default: internal periodic finite differences (2nd or 4th order).
- Optional: plug in your project derivative kernel via --dgamma module:function

Plugin contract for --dgamma:
- Called as: d6x, d6y, d6z = fn(gamma6, dx, **kwargs)
- gamma6 shape must be (6,N,N,N) in the order (g11,g22,g33,g12,g13,g23)

Outputs:
- L2 error for each direction (x,y,z) aggregated over all 6 components (or just diags)
- p_obs under refinement

This is the fastest way to decide:
- Christoffel assembly is fine (your test_2 suggests it is)
- So if p_obs ~ 0 here, the bug is in derivative scaling / packing / dx plumbing.
"""
import argparse
import importlib
import math
from typing import Callable, Tuple

import numpy as np
import scipy


# -----------------------------
# Grid + MMS
# -----------------------------

def make_grid(N: int, L: float):
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / float(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, dx

def omega(m: int, L: float) -> float:
    return 2.0 * math.pi * float(m) / float(L)

def build_gamma6_mms(N: int, L: float, eps: float, m: int) -> Tuple[np.ndarray, float]:
    X, Y, Z, dx = make_grid(N, L)
    w = omega(m, L)
    s = np.sin(w*X) * np.sin(w*Y) * np.sin(w*Z)
    g11 = 1.0 + eps*s
    g22 = 1.0 + eps*s
    g33 = 1.0 + eps*s
    g12 = np.zeros_like(g11)
    g13 = np.zeros_like(g11)
    g23 = np.zeros_like(g11)
    gamma6 = np.stack([g11,g22,g33,g12,g13,g23], axis=0)
    return gamma6, dx

def build_dgamma6_analytic(N: int, L: float, eps: float, m: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    X, Y, Z, _dx = make_grid(N, L)
    w = omega(m, L)
    sx = (w*np.cos(w*X)) * np.sin(w*Y) * np.sin(w*Z)
    sy = np.sin(w*X) * (w*np.cos(w*Y)) * np.sin(w*Z)
    sz = np.sin(w*X) * np.sin(w*Y) * (w*np.cos(w*Z))

    # Only diagonal components are nonzero; pack sym6 in canonical order.
    d6x = np.stack([eps*sx, eps*sx, eps*sx,
                    np.zeros_like(sx), np.zeros_like(sx), np.zeros_like(sx)], axis=0)
    d6y = np.stack([eps*sy, eps*sy, eps*sy,
                    np.zeros_like(sy), np.zeros_like(sy), np.zeros_like(sy)], axis=0)
    d6z = np.stack([eps*sz, eps*sz, eps*sz,
                    np.zeros_like(sz), np.zeros_like(sz), np.zeros_like(sz)], axis=0)
    return d6x, d6y, d6z


# -----------------------------
# Periodic finite differences
# -----------------------------

def d1_2nd(u: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(u, -1, axis=axis) - np.roll(u, 1, axis=axis)) / (2.0*dx)

def d1_4th(u: np.ndarray, dx: float, axis: int) -> np.ndarray:
    # 4th order central: ( -f_{i+2}+8f_{i+1}-8f_{i-1}+f_{i-2})/(12 dx)
    return (-np.roll(u, 2, axis=axis) + 8.0*np.roll(u, 1, axis=axis)
            - 8.0*np.roll(u, -1, axis=axis) + np.roll(u, -2, axis=axis)) / (12.0*dx)

def fd_dgamma6(gamma6: np.ndarray, dx: float, order: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    d1 = d1_2nd if order == 2 else d1_4th
    d6x = np.zeros_like(gamma6)
    d6y = np.zeros_like(gamma6)
    d6z = np.zeros_like(gamma6)
    for c in range(6):
        d6x[c] = d1(gamma6[c], dx, axis=0)
        d6y[c] = d1(gamma6[c], dx, axis=1)
        d6z[c] = d1(gamma6[c], dx, axis=2)
    return d6x, d6y, d6z


# -----------------------------
# Utilities
# -----------------------------

def l2(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a*a)))

def p_obs(E1: float, E2: float, h1: float, h2: float) -> float:
    return float(np.log(E1/E2)/np.log(h1/h2))

def load_plugin(spec: str) -> Callable:
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=str, default="16,32,48")
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=2, help="Integer Fourier mode (periodic by construction).")
    ap.add_argument("--order", type=int, choices=[2,4], default=2, help="FD derivative order (if no plugin).")
    ap.add_argument("--diag-only", action="store_true", help="Only measure error on diagonal sym6 comps (0,1,2).")
    ap.add_argument("--dgamma", type=str, default="", help="Plugin derivative: module:function (returns d6x,d6y,d6z).")
    ap.add_argument("--p-min", type=float, default=1.5, help="Minimum acceptable p_obs (2nd order default).")
    args = ap.parse_args()

    Ns = [int(s.strip()) for s in args.Ns.split(",") if s.strip()]
    plugin = load_plugin(args.dgamma) if args.dgamma else None
    print(f"[cfg] Ns={Ns} L={args.L} eps={args.eps} m={args.m} order={args.order} diag_only={args.diag_only}")
    if plugin:
        print(f"[cfg] plugin dgamma: {args.dgamma}")
    else:
        print("[cfg] using internal FD derivative")

    errors = []
    dxs = []

    for N in Ns:
        gamma6, dx = build_gamma6_mms(N, args.L, args.eps, args.m)
        d6x_ref, d6y_ref, d6z_ref = build_dgamma6_analytic(N, args.L, args.eps, args.m)

        if plugin:
            d6x_num, d6y_num, d6z_num = plugin(gamma6, dx)
        else:
            d6x_num, d6y_num, d6z_num = fd_dgamma6(gamma6, dx, args.order)

        comps = slice(0,3) if args.diag_only else slice(0,6)

        ex = l2((d6x_num[comps] - d6x_ref[comps]))
        ey = l2((d6y_num[comps] - d6y_ref[comps]))
        ez = l2((d6z_num[comps] - d6z_ref[comps]))
        e = math.sqrt((ex*ex + ey*ey + ez*ez)/3.0)

        # Off-diagonal sanity: these should be ~0 for this MMS
        off = slice(3,6)
        off_mag = l2(d6x_num[off]) + l2(d6y_num[off]) + l2(d6z_num[off])

        errors.append(e)
        dxs.append(dx)
        print(f"[run] N={N:4d}, dx={dx:.6f}, err={e:.6e} (ex={ex:.3e}, ey={ey:.3e}, ez={ez:.3e}), off_mag={off_mag:.3e}")

    # Convergence
    if len(Ns) >= 2:
        for i in range(len(Ns)-1):
            p = p_obs(errors[i], errors[i+1], dxs[i], dxs[i+1])
            print(f"[conv] p_obs({Ns[i]}→{Ns[i+1]}) = {p:.4f}")

    passed = True
    if len(Ns) >= 3:
        p01 = p_obs(errors[0], errors[1], dxs[0], dxs[1])
        p12 = p_obs(errors[1], errors[2], dxs[1], dxs[2])
        passed = (p01 >= args.p_min and p12 >= args.p_min)

    print({
        "passed": bool(passed),
        "metrics": {"errors": errors, "resolutions": Ns}
    })


if __name__ == "__main__":
    main()