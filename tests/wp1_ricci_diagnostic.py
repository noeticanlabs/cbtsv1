#!/usr/bin/env python3
"""
wp1_ricci_diagnostic.py
----------------------
Pinpoint *where* Ricci computation stops converging or disagrees with a trusted reference.

Layered checks:
  A) sym6 <-> full packing sanity
  B) dgamma (chosen discrete derivative) vs analytic MMS truth
  C) Christoffels: (optional) compiled kernel vs pure-Python assembly
  D) Ricci from (Gamma, dGamma) vs reference:
        - spectral FFT reference (periodic "truth")  [default]
        - python discrete reference (sanity; should match if using same Gamma)

Why this catches the p_obs ~ 0 bug:
- If dgamma converges (~2) but Ricci doesn't, your Ricci assembly or what you're comparing is wrong.
- If Python Ricci matches compiled Ricci but both disagree with spectral, your derivative scheme (e.g. np.gradient non-periodic) is the culprit.
- If compiled Gamma disagrees with Python Gamma, the compiled kernel or its input layout is wrong.

Run:
  python3 tests/wp1_ricci_diagnostic.py --Ns 16 32 48 --L 16 --eps 1e-3 --m 2 --crop 2
  python3 tests/wp1_ricci_diagnostic.py --Ns 16 32 48 --dgamma np_gradient --ref spectral --crop 2
  python3 tests/wp1_ricci_diagnostic.py --Ns 16 32 48 --use-compiled-christoffels --ref python

Exit status is not enforced; it prints JSON_SUMMARY for your CI / LoC gate.
"""

import argparse, json, math, sys
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# ----------------------------
# sym6 convention (match solver)
# [xx, xy, xz, yy, yz, zz]
# ----------------------------
IDX_XX, IDX_XY, IDX_XZ, IDX_YY, IDX_YZ, IDX_ZZ = 0, 1, 2, 3, 4, 5

def sym6_to_full(g6: np.ndarray) -> np.ndarray:
    g = np.zeros(g6.shape[:-1] + (3, 3), dtype=g6.dtype)
    g[..., 0, 0] = g6[..., IDX_XX]
    g[..., 0, 1] = g6[..., IDX_XY]
    g[..., 0, 2] = g6[..., IDX_XZ]
    g[..., 1, 0] = g6[..., IDX_XY]
    g[..., 1, 1] = g6[..., IDX_YY]
    g[..., 1, 2] = g6[..., IDX_YZ]
    g[..., 2, 0] = g6[..., IDX_XZ]
    g[..., 2, 1] = g6[..., IDX_YZ]
    g[..., 2, 2] = g6[..., IDX_ZZ]
    return g

def full_to_sym6(g: np.ndarray) -> np.ndarray:
    g6 = np.zeros(g.shape[:-2] + (6,), dtype=g.dtype)
    g6[..., IDX_XX] = g[..., 0, 0]
    g6[..., IDX_XY] = g[..., 0, 1]
    g6[..., IDX_XZ] = g[..., 0, 2]
    g6[..., IDX_YY] = g[..., 1, 1]
    g6[..., IDX_YZ] = g[..., 1, 2]
    g6[..., IDX_ZZ] = g[..., 2, 2]
    return g6

def crop_arr(a: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return a
    sl = (slice(crop, -crop), slice(crop, -crop), slice(crop, -crop))
    return a[sl]

def l2(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))

def l2_err(a: np.ndarray, b: np.ndarray, crop: int = 0) -> float:
    da = crop_arr(a, crop) - crop_arr(b, crop)
    return l2(da)

def p_obs(E1: float, E2: float, h1: float, h2: float) -> float:
    return float(np.log(E1 / E2) / np.log(h1 / h2))

# ----------------------------
# MMS metric: smooth periodic
# ----------------------------
def grid(N: int, L: float):
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = L / float(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, dx

def mms_gamma6(N: int, L: float, eps: float, m: int, offdiag: bool):
    X, Y, Z, dx = grid(N, L)
    w = 2.0 * math.pi * float(m) / float(L)
    s = np.sin(w * X) * np.sin(w * Y) * np.sin(w * Z)

    g_xx = 1.0 + eps * s
    g_yy = 1.0 + eps * s
    g_zz = 1.0 + eps * s

    if offdiag:
        g_xy = 0.2 * eps * np.sin(w * X) * np.sin(w * Y)
        g_xz = 0.2 * eps * np.sin(w * X) * np.sin(w * Z)
        g_yz = 0.2 * eps * np.sin(w * Y) * np.sin(w * Z)
    else:
        g_xy = 0.0 * s
        g_xz = 0.0 * s
        g_yz = 0.0 * s

    g6 = np.zeros((N, N, N, 6), dtype=np.float64)
    g6[..., IDX_XX] = g_xx
    g6[..., IDX_XY] = g_xy
    g6[..., IDX_XZ] = g_xz
    g6[..., IDX_YY] = g_yy
    g6[..., IDX_YZ] = g_yz
    g6[..., IDX_ZZ] = g_zz
    return g6, dx

def mms_dgamma6_truth(N: int, L: float, eps: float, m: int, offdiag: bool):
    X, Y, Z, _dx = grid(N, L)
    w = 2.0 * math.pi * float(m) / float(L)
    sx = (w * np.cos(w * X)) * np.sin(w * Y) * np.sin(w * Z)
    sy = np.sin(w * X) * (w * np.cos(w * Y)) * np.sin(w * Z)
    sz = np.sin(w * X) * np.sin(w * Y) * (w * np.cos(w * Z))

    d6x = np.zeros((N, N, N, 6), dtype=np.float64)
    d6y = np.zeros_like(d6x)
    d6z = np.zeros_like(d6x)

    for idx in (IDX_XX, IDX_YY, IDX_ZZ):
        d6x[..., idx] = eps * sx
        d6y[..., idx] = eps * sy
        d6z[..., idx] = eps * sz

    if offdiag:
        d6x[..., IDX_XY] = 0.2 * eps * (w * np.cos(w * X)) * np.sin(w * Y)
        d6y[..., IDX_XY] = 0.2 * eps * np.sin(w * X) * (w * np.cos(w * Y))

        d6x[..., IDX_XZ] = 0.2 * eps * (w * np.cos(w * X)) * np.sin(w * Z)
        d6z[..., IDX_XZ] = 0.2 * eps * np.sin(w * X) * (w * np.cos(w * Z))

        d6y[..., IDX_YZ] = 0.2 * eps * (w * np.cos(w * Y)) * np.sin(w * Z)
        d6z[..., IDX_YZ] = 0.2 * eps * np.sin(w * Y) * (w * np.cos(w * Z))

    return d6x, d6y, d6z

# ----------------------------
# Derivatives
# ----------------------------
def d1_periodic(u: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(u, -1, axis=axis) - np.roll(u, 1, axis=axis)) / (2.0 * dx)

def dgamma_fd_periodic(g6: np.ndarray, dx: float):
    d6x = np.zeros_like(g6)
    d6y = np.zeros_like(g6)
    d6z = np.zeros_like(g6)
    for c in range(6):
        d6x[..., c] = d1_periodic(g6[..., c], dx, axis=0)
        d6y[..., c] = d1_periodic(g6[..., c], dx, axis=1)
        d6z[..., c] = d1_periodic(g6[..., c], dx, axis=2)
    return d6x, d6y, d6z

def dgamma_np_gradient(g6: np.ndarray, dx: float):
    d6x = np.gradient(g6, dx, axis=0)
    d6y = np.gradient(g6, dx, axis=1)
    d6z = np.gradient(g6, dx, axis=2)
    return d6x, d6y, d6z

# ----------------------------
# Pure Python geometry assembly
# ----------------------------
def invert_metric(g: np.ndarray) -> np.ndarray:
    shp = g.shape
    G = g.reshape((-1, 3, 3))
    inv = np.linalg.inv(G)
    return inv.reshape(shp)

def christoffels_python(g: np.ndarray, dg: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dgx, dgy, dgz = dg
    invg = invert_metric(g)
    d = np.stack([dgx, dgy, dgz], axis=-1)  # (...,3,3,mu)
    Gamma = np.zeros(g.shape[:-2] + (3, 3, 3), dtype=np.float64)  # (..,k,i,j)

    for k in range(3):
        for i in range(3):
            for j in range(3):
                s = 0.0
                for ell in range(3):
                    term = d[..., j, ell, i] + d[..., i, ell, j] - d[..., i, j, ell]
                    s = s + invg[..., k, ell] * term
                Gamma[..., k, i, j] = 0.5 * s
    return Gamma

def dGamma_discrete(Gamma: np.ndarray, dx: float, scheme: str):
    if scheme == "periodic2":
        return (
            d1_periodic(Gamma, dx, axis=0),
            d1_periodic(Gamma, dx, axis=1),
            d1_periodic(Gamma, dx, axis=2),
        )
    if scheme == "np_gradient":
        return (
            np.gradient(Gamma, dx, axis=0),
            np.gradient(Gamma, dx, axis=1),
            np.gradient(Gamma, dx, axis=2),
        )
    raise ValueError("unknown dGamma scheme")

def ricci_from_Gamma_python(Gamma: np.ndarray, dGamma: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dGx, dGy, dGz = dGamma
    dG = np.stack([dGx, dGy, dGz], axis=-1)  # (...,k,i,j,mu)
    R = np.zeros(Gamma.shape[:-3] + (3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            term1 = 0.0
            term2 = 0.0
            term3 = 0.0
            term4 = 0.0
            for k in range(3):
                term1 = term1 + dG[..., k, i, j, k]  # ∂_k Γ^k_{ij}
                term2 = term2 + dG[..., k, i, k, j]  # ∂_j Γ^k_{ik}
                tr = 0.0
                for ell in range(3):
                    tr = tr + Gamma[..., ell, k, ell]
                term3 = term3 + Gamma[..., k, i, j] * tr
                for ell in range(3):
                    term4 = term4 + Gamma[..., ell, i, k] * Gamma[..., k, j, ell]
            R[..., i, j] = term1 - term2 + term3 - term4
    return R

# ----------------------------
# Spectral reference (periodic truth)
# ----------------------------
def spectral_derivative(f: np.ndarray, L: float, axis: int) -> np.ndarray:
    N = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    shape = [1, 1, 1]
    shape[axis] = N
    k = k.reshape(shape)
    F = np.fft.fftn(f)
    dF = 1j * k * F
    return np.fft.ifftn(dF).real

def spectral_ricci(g: np.ndarray, L: float) -> np.ndarray:
    N = g.shape[0]
    dg = np.zeros((N, N, N, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            dg[..., i, j, 0] = spectral_derivative(g[..., i, j], L, axis=0)
            dg[..., i, j, 1] = spectral_derivative(g[..., i, j], L, axis=1)
            dg[..., i, j, 2] = spectral_derivative(g[..., i, j], L, axis=2)

    invg = invert_metric(g)
    Gamma = np.zeros((N, N, N, 3, 3, 3), dtype=np.float64)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                s = 0.0
                for ell in range(3):
                    term = dg[..., j, ell, i] + dg[..., i, ell, j] - dg[..., i, j, ell]
                    s = s + invg[..., k, ell] * term
                Gamma[..., k, i, j] = 0.5 * s

    dG = np.zeros((N, N, N, 3, 3, 3, 3), dtype=np.float64)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                dG[..., k, i, j, 0] = spectral_derivative(Gamma[..., k, i, j], L, axis=0)
                dG[..., k, i, j, 1] = spectral_derivative(Gamma[..., k, i, j], L, axis=1)
                dG[..., k, i, j, 2] = spectral_derivative(Gamma[..., k, i, j], L, axis=2)

    R = np.zeros((N, N, N, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            term1 = 0.0
            term2 = 0.0
            term3 = 0.0
            term4 = 0.0
            for k in range(3):
                term1 = term1 + dG[..., k, i, j, k]
                term2 = term2 + dG[..., k, i, k, j]
                tr = 0.0
                for ell in range(3):
                    tr = tr + Gamma[..., ell, k, ell]
                term3 = term3 + Gamma[..., k, i, j] * tr
                for ell in range(3):
                    term4 = term4 + Gamma[..., ell, i, k] * Gamma[..., k, j, ell]
            R[..., i, j] = term1 - term2 + term3 - term4
    return R

# ----------------------------
# Optional: compiled Christoffels
# ----------------------------
def try_import_compiled():
    try:
        from src.core.gr_geometry_nsc import compute_christoffels_compiled
        return compute_christoffels_compiled
    except Exception:
        return None

@dataclass
class Row:
    N: int
    dx: float
    err_pack: float
    err_dgamma: float
    err_Gamma_comp_vs_py: Optional[float]
    err_Ricci_vs_ref: float

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[16, 32, 48])
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--offdiag", action="store_true")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--dgamma", choices=["periodic2", "np_gradient"], default="periodic2")
    ap.add_argument("--ref", choices=["spectral", "python"], default="spectral")
    ap.add_argument("--use-compiled-christoffels", action="store_true")
    args = ap.parse_args()

    comp = try_import_compiled() if args.use_compiled_christoffels else None
    if args.use_compiled_christoffels and comp is None:
        print("[warn] compiled christoffels not importable; continuing without", file=sys.stderr)

    rows = []
    for N in args.Ns:
        g6, dx = mms_gamma6(N, args.L, args.eps, args.m, offdiag=args.offdiag)
        g = sym6_to_full(g6)

        # A) packing
        g6_round = full_to_sym6(sym6_to_full(g6))
        err_pack = l2_err(g6, g6_round, crop=0)

        # B) dgamma
        d6x_t, d6y_t, d6z_t = mms_dgamma6_truth(N, args.L, args.eps, args.m, offdiag=args.offdiag)
        if args.dgamma == "periodic2":
            d6x, d6y, d6z = dgamma_fd_periodic(g6, dx)
        else:
            d6x, d6y, d6z = dgamma_np_gradient(g6, dx)
        err_dg = max(
            l2_err(d6x, d6x_t, args.crop),
            l2_err(d6y, d6y_t, args.crop),
            l2_err(d6z, d6z_t, args.crop),
        )

        # Python Gamma from same discrete derivatives
        dgx = sym6_to_full(d6x); dgy = sym6_to_full(d6y); dgz = sym6_to_full(d6z)
        Gamma_py = christoffels_python(g, (dgx, dgy, dgz))

        # C) compiled Gamma (optional)
        err_Gamma = None
        Gamma_use = Gamma_py
        if comp is not None:
            try:
                Gamma_comp, _ = comp(g6, d6x, d6y, d6z)
                if Gamma_comp.shape == Gamma_py.shape:
                    err_Gamma = l2_err(Gamma_comp, Gamma_py, args.crop)
                    Gamma_use = Gamma_comp
                else:
                    err_Gamma = None
            except Exception as e:
                print(f"[warn] compiled call failed at N={N}: {e}", file=sys.stderr)

        # D) Ricci using chosen Gamma + chosen discrete dGamma
        dG = dGamma_discrete(Gamma_use, dx, args.dgamma)
        R_num = ricci_from_Gamma_python(Gamma_use, dG)

        if args.ref == "spectral":
            R_ref = spectral_ricci(g, args.L)
        else:
            dG_py = dGamma_discrete(Gamma_py, dx, args.dgamma)
            R_ref = ricci_from_Gamma_python(Gamma_py, dG_py)

        err_R = l2_err(R_num, R_ref, args.crop)

        rows.append(Row(N=N, dx=dx, err_pack=err_pack, err_dgamma=err_dg,
                        err_Gamma_comp_vs_py=err_Gamma, err_Ricci_vs_ref=err_R))

        extra = f"  err_Gamma(comp-vs-py)={err_Gamma:.3e}" if err_Gamma is not None else ""
        print(f"[N={N:4d}] dx={dx:.6f}  pack={err_pack:.3e}  err_dgamma={err_dg:.3e}  err_Ricci={err_R:.3e}{extra}")

    # convergence on err_Ricci
    errs = [r.err_Ricci_vs_ref for r in rows]
    hs = [r.dx for r in rows]
    p_list = []
    for i in range(len(rows) - 1):
        p_list.append(p_obs(errs[i], errs[i + 1], hs[i], hs[i + 1]))
    pmin = min(p_list) if p_list else float("nan")

    summary = {
        "config": vars(args),
        "metrics": {
            "N": [r.N for r in rows],
            "dx": [r.dx for r in rows],
            "err_pack": [r.err_pack for r in rows],
            "err_dgamma": [r.err_dgamma for r in rows],
            "err_Ricci": errs,
            "p_obs_Ricci": p_list,
            "p_min_Ricci": pmin,
            "err_Gamma_compiled_vs_py": [r.err_Gamma_comp_vs_py for r in rows],
        },
        "interpretation": {
            "if_err_dgamma_converges_but_err_Ricci_flat": "Ricci assembly or reference mismatch (common: calling BSSN Ricci but comparing to raw Ricci).",
            "if_python_ref_matches_but_spectral_ref_fails": "Derivative scheme mismatch (np.gradient is not periodic; use periodic2 or crop).",
            "if_compiled_Gamma_disagrees": "Compiled kernel input layout or kernel logic mismatch.",
        }
    }

    print("\nJSON_SUMMARY:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()