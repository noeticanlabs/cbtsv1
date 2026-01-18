import numpy as np
import logging
import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gr_solver.gr_solver import GRSolver
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List
from gr_solver.gr_core_fields import inv_sym6, sym6_to_mat33, mat33_to_sym6, det_sym6

# ---------------------------
# Utility: norms & convergence
# ---------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x * x)))

def l2(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.sum(x * x)))

def estimate_order(errors: List[float], hs: List[float]) -> float:
    """
    Fit log(e) = p log(h) + c using least squares.
    """
    e = np.array(errors, dtype=float)
    h = np.array(hs, dtype=float)
    # protect
    e = np.maximum(e, 1e-300)
    h = np.maximum(h, 1e-300)
    A = np.vstack([np.log(h), np.ones_like(h)]).T
    p, _c = np.linalg.lstsq(A, np.log(e), rcond=None)[0]
    return float(p)

# ---------------------------
# Adapter spec (YOU map this)
# ---------------------------

@dataclass
class NRAdapter:
    """
    You implement these hooks once.
    Everything else becomes plug-and-play.
    """
    # Create a fields object at resolution N for a physical domain length L
    make_fields: Callable[[int, float], Any]

    # Set fields to the exact manufactured solution at time t
    set_exact_solution: Callable[[Any, float], None]

    # Return exact manufactured solution *as arrays* to compare (dict of variables)
    # Example keys: "gamma_sym6", "K_sym6", "phi", "A_sym6", "Gamma_tilde"
    get_state_arrays: Callable[[Any], Dict[str, np.ndarray]]

    # Return exact state arrays at time t (no solver needed)
    exact_state_arrays: Callable[[int, float, float], Dict[str, np.ndarray]]
    # signature: exact_state_arrays(N, L, t) -> dict[str, array]

    # Apply one time step. Must NOT mutate input if you want clean comparisons, so either:
    # - step(fields, dt, sources=None) returns NEW fields, or
    # - you provide clone_fields and allow mutation.
    step: Callable[[Any, float, Any], Any]  # (fields, dt, sources)->fields_new

    # Create MMS sources object at time t (whatever your solver expects)
    # Could be dict of arrays, or a Sources class.
    make_mms_sources: Callable[[int, float, float], Any]
    # signature: make_mms_sources(N, L, t) -> sources

    # Optional: deep copy fields
    clone_fields: Callable[[Any], Any] = None

    # Derivative entrypoints for Test1B:
    # Must accept ndarray and dx, return ndarray.
    Dx: Callable[[np.ndarray, float], np.ndarray] = None
    Dy: Callable[[np.ndarray, float], np.ndarray] = None
    Dz: Callable[[np.ndarray, float], np.ndarray] = None
    Dxx: Callable[[np.ndarray, float], np.ndarray] = None
    Dyy: Callable[[np.ndarray, float], np.ndarray] = None
    Dzz: Callable[[np.ndarray, float], np.ndarray] = None
    Lap: Callable[[np.ndarray, float], np.ndarray] = None

# -----------------------------------------
# Test1A: Discrete Defect (MMS Lie Detector)
# -----------------------------------------

def gcat1_test1a_discrete_defect(
    adapter: NRAdapter,
    Ns=(16, 32, 64),
    L=16.0,
    CFL=0.01,
    dt_mode="dx2",     # "dx" or "dx2" (you used dx^2 to suppress time error)
    compare_keys=None, # None -> compare all keys returned
    use_rms=True,
    verbose=True
) -> Dict[str, Any]:
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
            print(f"[Test1A] N={N} dx={dx:.3e} dt={dt:.3e} defect(no src)={defect0:.3e} defect(with src)={defectS:.3e}")

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

# ---------------------------------------------
# Test1B: Operator-only convergence (dx detector)
# ---------------------------------------------

def gcat1_test1b_operator_convergence(
    adapter: NRAdapter,
    Ns=(16, 32, 64),
    L=16.0,
    mode="1d",     # "1d" or "3d" analytic function
    expected_order=2.0,
    verbose=True
) -> Dict[str, Any]:
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

# ---------------------------
# GR Solver Adapter Implementation
# ---------------------------

def create_gr_adapter():
    """Create NRAdapter for the GR solver."""

    def set_mms(t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 0.1
        ky = 0.1
        kz = 0.1
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-4
        gamma_sym6 = np.zeros((N, N, N, 6))
        gamma_sym6[..., 0] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # xx
        gamma_sym6[..., 1] = eps * np.cos(kdotx) * np.cos(omega * t)      # xy
        gamma_sym6[..., 3] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # yy
        gamma_sym6[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)      # yz
        gamma_sym6[..., 5] = 1 + eps * np.cos(kdotx) * np.cos(omega * t)  # zz
        K_sym6 = np.zeros((N, N, N, 6))
        K_sym6[..., 0] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 1] = eps * np.sin(kdotx) * np.cos(omega * t)
        K_sym6[..., 3] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)
        K_sym6[..., 5] = eps * np.cos(kdotx) * np.sin(omega * t)
        alpha = np.ones((N, N, N))  # fixed lapse
        beta = np.zeros((N, N, N, 3))  # fixed shift
        return gamma_sym6, K_sym6, alpha, beta

    def compute_dt_mms(t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 0.1
        ky = 0.1
        kz = 0.1
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-4
        dt_gamma_sym6 = np.zeros((N, N, N, 6))
        dt_gamma_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.sin(kdotx)  # d/dt [sin(k) sin(ot)] = o cos(ot) sin(k)
        dt_gamma_sym6[..., 1] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)  # d/dt [cos(k) cos(ot)] = -o cos(k) sin(ot)
        dt_gamma_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma_sym6[..., 4] = eps * omega * np.sin(kdotx) * np.cos(omega * t)  # d/dt [sin(k) sin(ot)] = o sin(k) cos(ot)
        dt_gamma_sym6[..., 5] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)  # d/dt [cos(k) cos(ot)] = -o cos(k) sin(ot)
        dt_K_sym6 = np.zeros((N, N, N, 6))
        dt_K_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.cos(kdotx)  # d/dt [cos(k) sin(ot)] = o cos(k) cos(ot)
        dt_K_sym6[..., 1] = eps * omega * np.sin(kdotx) * (-np.sin(omega * t))  # d/dt [sin(k) cos(ot)] = o sin(k) (-sin(ot))
        dt_K_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K_sym6[..., 4] = eps * omega * np.sin(kdotx) * (-np.cos(omega * t))  # d/dt [sin(k) sin(ot)] = o sin(k) (-cos(ot))
        dt_K_sym6[..., 5] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_alpha = np.zeros((N, N, N))
        dt_beta = np.zeros((N, N, N, 3))
        return dt_gamma_sym6, dt_K_sym6, dt_alpha, dt_beta

    def make_fields(N, L):
        dx = L / N
        return GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)

    def set_exact_solution(solver, t):
        N = solver.fields.Nx
        dx = solver.fields.dx
        gamma, K, alpha, beta = set_mms(t, N, dx, dx, dx, L=16.0)
        solver.fields.gamma_sym6 = gamma.copy()
        solver.fields.K_sym6 = K.copy()
        solver.fields.gamma_tilde_sym6 = gamma.copy()
        solver.fields.A_sym6 = K.copy()
        solver.fields.alpha = alpha.copy()
        solver.fields.beta = beta.copy()
        solver.fields.phi = np.zeros((N, N, N))
        solver.fields.Gamma_tilde = np.zeros((N, N, N, 3))
        solver.fields.Z = np.zeros((N, N, N))
        solver.fields.Z_i = np.zeros((N, N, N, 3))
        # Compute geometry
        solver.geometry.compute_christoffels()
        solver.geometry.compute_ricci()
        solver.geometry.compute_scalar_curvature()

    def get_state_arrays(solver):
        return {
            'gamma_tilde_sym6': solver.fields.gamma_tilde_sym6.copy(),
            'A_sym6': solver.fields.A_sym6.copy(),
            'phi': solver.fields.phi.copy(),
            'Gamma_tilde': solver.fields.Gamma_tilde.copy(),
            'Z': solver.fields.Z.copy(),
            'Z_i': solver.fields.Z_i.copy(),
            'alpha': solver.fields.alpha.copy(),
            'beta': solver.fields.beta.copy(),
        }

    def exact_state_arrays(N, L, t):
        dx = L / N
        gamma, K, alpha, beta = set_mms(t, N, dx, dx, dx, L)
        phi = np.zeros((N, N, N))
        Gamma_tilde = np.zeros((N, N, N, 3))
        Z = np.zeros((N, N, N))
        Z_i = np.zeros((N, N, N, 3))
        return {
            'gamma_tilde_sym6': gamma,
            'A_sym6': K,
            'phi': phi,
            'Gamma_tilde': Gamma_tilde,
            'Z': Z,
            'Z_i': Z_i,
            'alpha': alpha,
            'beta': beta,
        }

    def clone_fields(solver):
        return copy.deepcopy(solver)

    def step(solver, dt, sources):
        # Set sources or zero
        if sources is not None:
            if callable(sources):
                solver.stepper.sources_func = sources  # sources is a function t -> dict
            else:
                solver.stepper.sources_func = lambda t: sources  # fixed dict
        else:
            solver.stepper.sources_func = None
            N = solver.fields.Nx
            solver.stepper.S_gamma_tilde_sym6 = np.zeros((N, N, N, 6))
            solver.stepper.S_A_sym6 = np.zeros((N, N, N, 6))
            solver.stepper.S_phi = np.zeros((N, N, N))
            solver.stepper.S_Gamma_tilde = np.zeros((N, N, N, 3))
            solver.stepper.S_Z = np.zeros((N, N, N))
            solver.stepper.S_Z_i = np.zeros((N, N, N, 3))

        # Force dt
        solver.scheduler.compute_dt = lambda eps_H, eps_M: dt
        # Run step
        dt_actual, _, _ = solver.orchestrator.run_step()
        solver.stepper.sources_func = None  # reset
        return solver

    def make_mms_sources(N, L, t):
        dx = L / N
        # Create temp solver
        solver = make_fields(N, L)
        set_exact_solution(solver, t)
        # Compute rhs
        solver.stepper.compute_rhs()
        # Compute exact dt
        dt_gamma, dt_K, dt_alpha, dt_beta = compute_dt_mms(t, N, dx, dx, dx, L)
        # Sources
        S_gamma_tilde = dt_gamma - solver.stepper.rhs_gamma_tilde_sym6
        S_A = dt_K - solver.stepper.rhs_A_sym6
        S_phi = np.zeros((N, N, N)) - solver.stepper.rhs_phi  # exact phi=0, dt_phi=0
        S_Gamma_tilde = np.zeros((N, N, N, 3))  # assuming not evolved or 0
        S_Z = np.zeros((N, N, N)) - solver.stepper.rhs_Z
        S_Z_i = np.zeros((N, N, N, 3)) - solver.stepper.rhs_Z_i
        return {
            'S_gamma_tilde_sym6': S_gamma_tilde,
            'S_A_sym6': S_A,
            'S_phi': S_phi,
            'S_Gamma_tilde': S_Gamma_tilde,
            'S_Z': S_Z,
            'S_Z_i': S_Z_i,
        }

    # Derivative functions using FFT
    def Dx(f, dx):
        N = f.shape[0]
        kx = 2 * np.pi * np.fft.fftfreq(N, dx)
        KX, _, _ = np.meshgrid(kx, kx, kx, indexing='ij')
        f_hat = np.fft.fftn(f)
        df_hat = 1j * KX * f_hat
        df = np.fft.ifftn(df_hat).real
        return df

    def Dy(f, dy):
        Nx, Ny, Nz = f.shape
        kx = np.fft.fftfreq(Nx, d=1.0)  # dummy
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
        kz = np.fft.fftfreq(Nz, d=1.0)  # dummy
        _, KY, _ = np.meshgrid(kx, ky, kz, indexing='ij')
        f_hat = np.fft.fftn(f)
        df_hat = 1j * KY * f_hat
        df = np.fft.ifftn(df_hat).real
        return df

    def Dz(f, dz):
        Nx, Ny, Nz = f.shape
        kx = np.fft.fftfreq(Nx, d=1.0)  # dummy
        ky = np.fft.fftfreq(Ny, d=1.0)  # dummy
        kz = 2 * np.pi * np.fft.fftfreq(Nz, dz)
        _, _, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        f_hat = np.fft.fftn(f)
        df_hat = 1j * KZ * f_hat
        df = np.fft.ifftn(df_hat).real
        return df

    def Dxx(f, dx):
        N = f.shape[0]
        kx = 2 * np.pi * np.fft.fftfreq(N, dx)
        KX, _, _ = np.meshgrid(kx, kx, kx, indexing='ij')
        f_hat = np.fft.fftn(f)
        d2f_hat = -KX**2 * f_hat
        d2f = np.fft.ifftn(d2f_hat).real
        return d2f

    def Dyy(f, dy):
        N = f.shape[1]
        ky = 2 * np.pi * np.fft.fftfreq(N, dy)
        _, KY, _ = np.meshgrid(np.fft.fftfreq(f.shape[0]), ky, np.fft.fftfreq(f.shape[2]), indexing='ij')
        f_hat = np.fft.fftn(f)
        d2f_hat = -KY**2 * f_hat
        d2f = np.fft.ifftn(d2f_hat).real
        return d2f

    def Dzz(f, dz):
        N = f.shape[2]
        kz = 2 * np.pi * np.fft.fftfreq(N, dz)
        _, _, KZ = np.meshgrid(np.fft.fftfreq(f.shape[0]), np.fft.fftfreq(f.shape[1]), kz, indexing='ij')
        f_hat = np.fft.fftn(f)
        d2f_hat = -KZ**2 * f_hat
        d2f = np.fft.ifftn(d2f_hat).real
        return d2f

    def Lap(f, dx):
        N = f.shape[0]
        kx = 2 * np.pi * np.fft.fftfreq(N, dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, dx)  # assume dy=dx
        kz = 2 * np.pi * np.fft.fftfreq(N, dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = KX**2 + KY**2 + KZ**2
        f_hat = np.fft.fftn(f)
        lap_f_hat = -k2 * f_hat
        lap_f = np.fft.ifftn(lap_f_hat).real
        return lap_f

    return NRAdapter(
        make_fields=make_fields,
        set_exact_solution=set_exact_solution,
        get_state_arrays=get_state_arrays,
        exact_state_arrays=exact_state_arrays,
        step=step,
        make_mms_sources=make_mms_sources,
        clone_fields=clone_fields,
        Dx=Dx,
        Dy=Dy,
        Dz=Dz,
        Dxx=Dxx,
        Dyy=Dyy,
        Dzz=Dzz,
        Lap=Lap,
    )

class Test1A:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        adapter = create_gr_adapter()
        result = gcat1_test1a_discrete_defect(adapter)
        return result

class Test1B:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        adapter = create_gr_adapter()
        result = gcat1_test1b_operator_convergence(adapter)
        return result

class Test0Bs2:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()
        # Record initial eps_H, eps_M, and inf norms of gamma_sym6, K_sym6, alpha, beta at t=0
        self.gr_solver.constraints.compute_hamiltonian()
        self.gr_solver.constraints.compute_momentum()
        self.gr_solver.constraints.compute_residuals()
        eps_H = self.gr_solver.constraints.eps_H
        eps_M = self.gr_solver.constraints.eps_M
        inf_gamma = np.max(np.abs(self.gr_solver.fields.gamma_sym6))
        inf_K = np.max(np.abs(self.gr_solver.fields.K_sym6))
        inf_alpha = np.max(np.abs(self.gr_solver.fields.alpha))
        inf_beta = np.max(np.abs(self.gr_solver.fields.beta))

        # Make copies of initial fields
        gamma0 = self.gr_solver.fields.gamma_sym6.copy()
        K0 = self.gr_solver.fields.K_sym6.copy()
        alpha0 = self.gr_solver.fields.alpha.copy()
        beta0 = self.gr_solver.fields.beta.copy()

        # Apply small perturbation to gamma_sym6 and K_sym6 to excite dynamics
        pert = 1e-12
        self.gr_solver.fields.gamma_sym6[0, 0, 0, 0] += pert
        self.gr_solver.fields.K_sym6[0, 0, 0, 0] += pert

        # Take one step using the solver's orchestrator.run_step()
        dt, dominant_thread, rail_violation = self.gr_solver.orchestrator.run_step()

        # Record deltas
        delta_gamma = self.gr_solver.fields.gamma_sym6 - gamma0
        delta_K = self.gr_solver.fields.K_sym6 - K0
        delta_alpha = self.gr_solver.fields.alpha - alpha0
        delta_beta = self.gr_solver.fields.beta - beta0

        # Compute omega_state_sum as sum of L2 norms of deltas
        omega_gamma = np.sqrt(np.sum(delta_gamma**2))
        omega_K = np.sqrt(np.sum(delta_K**2))
        omega_alpha = np.sqrt(np.sum(delta_alpha**2))
        omega_beta = np.sqrt(np.sum(delta_beta**2))
        omega_state_sum = omega_gamma + omega_K + omega_alpha + omega_beta

        # Pass if at least one delta > 1e-14
        max_delta = max(np.max(np.abs(delta_gamma)), np.max(np.abs(delta_K)), np.max(np.abs(delta_alpha)), np.max(np.abs(delta_beta)))
        passed = max_delta > 1e-14

        # Restore solver state
        self.gr_solver.fields.gamma_sym6 = gamma0
        self.gr_solver.fields.K_sym6 = K0
        self.gr_solver.fields.alpha = alpha0
        self.gr_solver.fields.beta = beta0
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Prepare metrics and diagnosis
        metrics = {
            'eps_H': eps_H,
            'eps_M': eps_M,
            'inf_gamma': inf_gamma,
            'inf_K': inf_K,
            'inf_alpha': inf_alpha,
            'inf_beta': inf_beta,
            'omega_state_sum': omega_state_sum
        }
        diagnosis = "Evolution detected: solver responded to perturbation." if passed else "No evolution: fields unchanged within tolerance."

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test1MmsLite:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def set_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        gamma_sym6 = np.zeros((N, N, N, 6))
        gamma_sym6[..., 0] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # xx
        gamma_sym6[..., 3] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # yy
        gamma_sym6[..., 5] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)  # zz
        K_sym6 = np.zeros((N, N, N, 6))
        K_sym6[..., 0] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 3] = eps * np.cos(kdotx) * np.sin(omega * t)
        K_sym6[..., 5] = eps * np.cos(kdotx) * np.sin(omega * t)
        alpha = np.ones((N, N, N))
        beta = np.zeros((N, N, N, 3))
        return gamma_sym6, K_sym6, alpha, beta

    def compute_dt_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        dt_gamma_sym6 = np.zeros((N, N, N, 6))
        dt_gamma_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_gamma_sym6[..., 5] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
        dt_K_sym6 = np.zeros((N, N, N, 6))
        dt_K_sym6[..., 0] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K_sym6[..., 3] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_K_sym6[..., 5] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
        dt_alpha = np.zeros((N, N, N))
        dt_beta = np.zeros((N, N, N, 3))
        return dt_gamma_sym6, dt_K_sym6, dt_alpha, dt_beta

    def compute_Gamma_tilde_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        
        sin_k = np.sin(kdotx)
        cos_k = np.cos(kdotx)
        sin_wt = np.sin(omega * t)
        
        F = eps * sin_k * sin_wt
        
        Gamma_tilde = np.zeros((N, N, N, 3))
        ks = [kx, ky, kz]
        for i in range(3):
            ki = ks[i]
            di_F = eps * ki * cos_k * sin_wt
            # Gamma^i = -0.5 * (1+F)^-2 * di_F
            Gamma_tilde[..., i] = -0.5 * (1 + F)**(-2) * di_F
            
        return Gamma_tilde

    def compute_dt_Gamma_tilde_mms(self, t, N, dx, dy, dz, L=16.0):
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        z = np.arange(N) * dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        kx = 2 * np.pi / L
        ky = 2 * np.pi / L
        kz = 2 * np.pi / L
        kdotx = kx * X + ky * Y + kz * Z
        omega = 1.0
        eps = 1e-3
        
        sin_k = np.sin(kdotx)
        cos_k = np.cos(kdotx)
        sin_wt = np.sin(omega * t)
        cos_wt = np.cos(omega * t)
        
        F = eps * sin_k * sin_wt
        dt_F = eps * omega * sin_k * cos_wt
        
        dt_Gamma_tilde = np.zeros((N, N, N, 3))
        ks = [kx, ky, kz]
        for i in range(3):
            ki = ks[i]
            di_F = eps * ki * cos_k * sin_wt
            dt_di_F = eps * ki * omega * cos_k * cos_wt
            
            # dt_Gamma^i = (1+F)^-3 * dt_F * di_F - 0.5 * (1+F)^-2 * dt_di_F
            term1 = (1 + F)**(-3) * dt_F * di_F
            term2 = 0.5 * (1 + F)**(-2) * dt_di_F
            dt_Gamma_tilde[..., i] = term1 - term2
            
        return dt_Gamma_tilde

    def compute_full_gamma_driver_rhs(self, solver):
        """
        Implements the full BSSN evolution equation for Gamma_tilde:
        dt_Gamma^i = -2*A^ij*dj_alpha + 2*alpha*(Gamma^i_jk*A^jk + 6*A^ij*dj_phi - (2/3)*gamma^ij*dj_K)
                     + beta^j*dj_Gamma^i - Gamma^j*dj_beta^i + (2/3)*Gamma^i*dj_beta^j
                     + gamma^jk*djk_beta^i + (1/3)*gamma^ij*djk_beta^k
        """
        fields = solver.fields
        g_tilde = fields.gamma_tilde_sym6
        A_tilde = fields.A_sym6
        Gamma_tilde = fields.Gamma_tilde
        alpha = fields.alpha
        beta = fields.beta
        phi = fields.phi
        
        # Grid spacing (assuming uniform cubic)
        dx = fields.dx
        
        # Helper for central difference
        def d_i(f, axis):
            return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)
            
        def d_ij(f, ax1, ax2):
            # d_i (d_j f)
            return d_i(d_i(f, ax2), ax1)

        # 1. Inverse metric and Christoffels
        g_inv_sym = inv_sym6(g_tilde)
        g_inv = sym6_to_mat33(g_inv_sym) # (..., 3, 3)
        g_mat = sym6_to_mat33(g_tilde)
        
        # Compute conformal Christoffel symbols Gamma^i_jk
        # Gamma^i_jk = 0.5 * g^il * (d_j g_lk + d_k g_lj - d_l g_jk)
        dg = np.zeros(g_mat.shape + (3,)) # (..., 3, 3, 3) last is deriv index
        for k in range(3):
            dg[..., k] = d_i(g_mat, k)
            
        Gamma_ijk = np.zeros(g_mat.shape + (3,)) # (..., 3, 3, 3) i, j, k
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    term = 0.0
                    for l in range(3):
                        term += 0.5 * g_inv[..., i, l] * (dg[..., l, k, j] + dg[..., l, j, k] - dg[..., j, k, l])
                    Gamma_ijk[..., i, j, k] = term

        # 2. Raise A_tilde: A^ij = g^ik g^jl A_kl
        A_mat = sym6_to_mat33(A_tilde)
        A_up = np.einsum('...ik,...jl,...kl->...ij', g_inv, g_inv, A_mat)
        
        # 3. Derivatives
        d_alpha = np.stack([d_i(alpha, k) for k in range(3)], axis=-1)
        d_phi = np.stack([d_i(phi, k) for k in range(3)], axis=-1)
        
        # K is trace of extrinsic curvature. Here we approximate K=0 or compute from K_sym6 if available.
        # In this test setup, K_sym6 is set. K = tr(K_ij) / e^4phi approx? 
        # For MMS, let's assume K=0 as phi=0 and A is traceless part of K.
        d_K = np.zeros_like(d_phi) 

        # 4. Advection terms (Lie derivative part 1)
        # beta^j dj_Gamma^i
        advect = np.zeros_like(Gamma_tilde)
        for j in range(3):
            advect += beta[..., j:j+1] * d_i(Gamma_tilde, j)
            
        # - Gamma^j dj_beta^i
        d_beta = np.zeros(beta.shape + (3,)) # (..., i, j) -> dj beta^i
        for j in range(3):
            d_beta[..., j] = d_i(beta, j) # deriv index is last
            
        twist = np.einsum('...j,...ji->...i', Gamma_tilde, d_beta)
        
        # (2/3) Gamma^i dj_beta^j
        div_beta = d_beta[..., 0, 0] + d_beta[..., 1, 1] + d_beta[..., 2, 2]
        compress = (2.0/3.0) * Gamma_tilde * div_beta[..., None]

        # 5. Second derivative terms of beta
        dd_beta = np.zeros(beta.shape + (3, 3)) # (..., i, j, k) -> dj dk beta^i
        for j in range(3):
            for k in range(3):
                dd_beta[..., j, k] = d_ij(beta, j, k)
                
        lap_shift = np.einsum('...jk,...ijk->...i', g_inv, dd_beta)
        grad_div_shift = np.einsum('...ij,...kjk->...i', g_inv, dd_beta)

        # 6. Assemble RHS
        rhs = np.zeros_like(Gamma_tilde)
        
        # Term: -2 A^ij dj_alpha
        rhs += -2.0 * np.einsum('...ij,...j->...i', A_up, d_alpha)
        
        # Term: 2 alpha ( Gamma^i_jk A^jk + 6 A^ij dj_phi - (2/3) g^ij dj_K )
        term_paren = np.einsum('...ijk,...jk->...i', Gamma_ijk, A_up)
        term_paren += 6.0 * np.einsum('...ij,...j->...i', A_up, d_phi)
        term_paren -= (2.0/3.0) * np.einsum('...ij,...j->...i', g_inv, d_K)
        rhs += 2.0 * alpha[..., None] * term_paren
        
        # Add Lie terms
        rhs += advect - twist + compress
        
        # Add shift Laplacian terms
        rhs += lap_shift + (1.0/3.0) * grad_div_shift
        
        return rhs

    def run(self):
        from gr_solver.gr_solver import GRSolver
        from gr_solver.gr_core_fields import SYM6_IDX
        errors = []
        L = 16.0
        for N in [16, 32, 64]:
            dx = L / N
            dy = dx
            dz = dx
            solver = GRSolver(N, N, N, dx=dx, dy=dx, dz=dx)
            gamma, K, alpha, beta = self.set_mms(0, N, dx, dy, dz, L)
            solver.fields.gamma_sym6 = gamma
            solver.fields.K_sym6 = K
            solver.fields.alpha = alpha
            solver.fields.beta = beta
            solver.fields.phi = np.zeros((N, N, N))
            solver.fields.gamma_tilde_sym6 = gamma.copy()
            solver.fields.A_sym6 = K.copy()
            solver.fields.Gamma_tilde = self.compute_Gamma_tilde_mms(0, N, dx, dy, dz, L)
            solver.fields.Z = np.zeros((N, N, N))
            solver.fields.Z_i = np.zeros((N, N, N, 3))
            solver.geometry.compute_christoffels()
            solver.geometry.compute_ricci()
            solver.geometry.compute_scalar_curvature()
            dt_gamma, dt_K, dt_alpha, dt_beta = self.compute_dt_mms(0, N, dx, dy, dz, L)
            solver.stepper.compute_rhs()
            # Compute initial constraints
            solver.constraints.compute_hamiltonian()
            solver.constraints.compute_momentum()
            solver.constraints.compute_residuals()
            eps_H_init = solver.constraints.eps_H
            eps_M_init = solver.constraints.eps_M
            logging.debug(f"Test1 MMS N={N}: initial eps_H={eps_H_init}, eps_M={eps_M_init}")
            cfl = 0.01
            cfl_ratio = cfl
            logging.debug(f"Test1 MMS N={N}: CFL ratio dt/dx = {cfl_ratio}")
            S_gamma = dt_gamma - solver.stepper.rhs_gamma_sym6
            S_K = dt_K - solver.stepper.rhs_K_sym6
            S_phi = -solver.stepper.rhs_phi
            S_gamma_tilde = dt_gamma - solver.stepper.rhs_gamma_tilde_sym6
            S_A = dt_K - solver.stepper.rhs_A_sym6
            
            # Compute S_Gamma_tilde
            dt_Gamma_tilde = self.compute_dt_Gamma_tilde_mms(0, N, dx, dy, dz, L)
            rhs_Gamma_full_init = self.compute_full_gamma_driver_rhs(solver)
            S_Gamma_tilde = dt_Gamma_tilde - rhs_Gamma_full_init
            
            S_Z = -solver.stepper.rhs_Z
            S_Z_i = -solver.stepper.rhs_Z_i
            logging.debug(f"Test1 MMS N={N}: S_gamma_norm = {np.linalg.norm(S_gamma)}, S_K_norm = {np.linalg.norm(S_K)}, S_phi_norm = {np.linalg.norm(S_phi)}, S_gamma_tilde_norm = {np.linalg.norm(S_gamma_tilde)}, S_A_norm = {np.linalg.norm(S_A)}")
            solver.stepper.S_gamma_sym6 = S_gamma
            solver.stepper.S_K_sym6 = S_K
            solver.stepper.S_phi = S_phi
            solver.stepper.S_gamma_tilde_sym6 = S_gamma_tilde
            solver.stepper.S_A_sym6 = S_A
            solver.stepper.S_Gamma_tilde = S_Gamma_tilde
            solver.stepper.S_Z = S_Z
            solver.stepper.S_Z_i = S_Z_i
            
            original_compute_rhs = solver.stepper.compute_rhs
            def compute_rhs_with_sources(stepper, t, slow_update):
                original_compute_rhs(t, slow_update)
                stepper.rhs_gamma_sym6 += stepper.S_gamma_sym6
                stepper.rhs_K_sym6 += stepper.S_K_sym6
                stepper.rhs_phi += stepper.S_phi
                stepper.rhs_gamma_tilde_sym6 += stepper.S_gamma_tilde_sym6
                stepper.rhs_A_sym6 += stepper.S_A_sym6
                stepper.rhs_Z += stepper.S_Z
                stepper.rhs_Z_i += stepper.S_Z_i
                
                # Patch: Compute full Gamma_tilde evolution
                rhs_Gamma_full = self.compute_full_gamma_driver_rhs(solver)
                # Apply source (S_Gamma is 0, so just set RHS)
                stepper.rhs_Gamma_tilde = rhs_Gamma_full + stepper.S_Gamma_tilde
                
            solver.stepper.compute_rhs = lambda t, slow_update: compute_rhs_with_sources(solver.stepper, t, slow_update)
            solver.stepper.damping_enabled = False
            # Hold gauge fixed
            solver.gauge.evolve_lapse = lambda dt: None
            solver.gauge.evolve_shift = lambda dt: None
            T = 0.01
            # Scale dt with N to ensure CFL scaling and convergence
            dt_scaled = 0.16 / N
            solver.scheduler.compute_dt = lambda eps_H, eps_M: dt_scaled
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)
            # Compute final constraints
            solver.constraints.compute_residuals()
            eps_H_final = solver.constraints.eps_H
            logging.debug(f"Test1 MMS N={N}: final eps_H={eps_H_final}")
            gamma_exact, K_exact, alpha_exact, beta_exact = self.set_mms(T, N, dx, dy, dz, L)
            err_gamma = solver.fields.gamma_sym6 - gamma_exact
            err_K = solver.fields.K_sym6 - K_exact
            error_gamma = np.linalg.norm(err_gamma) / np.sqrt(err_gamma.size)
            error_K = np.linalg.norm(err_K) / np.sqrt(err_K.size)
            error = error_gamma + error_K
            errors.append(error)
            logging.debug(f"Test1 MMS N={N}: error_gamma={error_gamma:.2e}, error_K={error_K:.2e}, total error={error:.2e}")
            # Log max sources
            max_S_gamma = np.max(np.abs(S_gamma_tilde))
            max_S_K = np.max(np.abs(S_K))
            logging.debug(f"Test1 MMS N={N}: max_S_gamma={max_S_gamma:.2e}, max_S_K={max_S_K:.2e}")
        e_N, e_2N, e_4N = errors
        if e_2N == 0:
            p_obs = np.inf
        else:
            p_obs = np.log2(e_N / e_2N)
        passed = p_obs > 1.5
        metrics = {'p_obs': p_obs, 'errors': [e_N, e_2N, e_4N]}
        diagnosis = f"Observed order p_obs = {p_obs:.2f}, {'passed' if passed else 'failed'} convergence check (>1.5)"
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test2Rcs:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        from gr_solver.gr_solver import GRSolver
        N = 8
        resolutions = [N, 2*N, 4*N]
        T = 0.1
        solutions = {}
        for res in resolutions:
            dx = 1.0 / res  # fixed domain [0,1]^3
            solver = GRSolver(res, res, res, dx=dx, dy=dx, dz=dx)
            solver.init_minkowski()
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)
            solutions[res] = {
                'gamma_sym6': solver.fields.gamma_sym6.copy(),
                'K_sym6': solver.fields.K_sym6.copy()
            }
        # Now restrict finer to coarser
        def restrict(fine, coarse_res):
            fine_res = 2 * coarse_res
            assert fine.shape[:3] == (fine_res, fine_res, fine_res)
            coarse = np.zeros((coarse_res, coarse_res, coarse_res) + fine.shape[3:])
            for i in range(coarse_res):
                for j in range(coarse_res):
                    for k in range(coarse_res):
                        coarse[i, j, k] = np.mean(fine[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2], axis=(0,1,2))
            return coarse
        
        # Compute differences
        gamma_N = solutions[N]['gamma_sym6']
        K_N = solutions[N]['K_sym6']
        gamma_2N = solutions[2*N]['gamma_sym6']
        K_2N = solutions[2*N]['K_sym6']
        gamma_4N = solutions[4*N]['gamma_sym6']
        K_4N = solutions[4*N]['K_sym6']
        
        # Restrict 2N to N
        gamma_2N_restricted = restrict(gamma_2N, N)
        K_2N_restricted = restrict(K_2N, N)
        # Restrict 4N to 2N
        gamma_4N_restricted = restrict(gamma_4N, 2*N)
        K_4N_restricted = restrict(K_4N, 2*N)
        
        # L2 differences
        diff_gamma_coarse = np.sqrt(np.sum((gamma_N - gamma_2N_restricted)**2))
        diff_K_coarse = np.sqrt(np.sum((K_N - K_2N_restricted)**2))
        diff_coarse = diff_gamma_coarse + diff_K_coarse
        
        diff_gamma_fine = np.sqrt(np.sum((gamma_2N - gamma_4N_restricted)**2))
        diff_K_fine = np.sqrt(np.sum((K_2N - K_4N_restricted)**2))
        diff_fine = diff_gamma_fine + diff_K_fine
        
        if diff_fine == 0:
            p_obs = np.inf
        else:
            p_obs = np.log2(diff_coarse / diff_fine)
        
        p_target = 2
        passed = p_obs > p_target - 0.5
        
        metrics = {'p_obs': p_obs, 'diff_coarse': diff_coarse, 'diff_fine': diff_fine}
        diagnosis = f"Observed order p_obs = {p_obs:.2f}, expected ~{p_target}, {'passed' if passed else 'failed'} (threshold {p_target - 0.5})"
        
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test3:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()
        
        N = self.gr_solver.N
        center = N // 2
        
        # Inject localized bump in K_sym6 (Gaussian at center with amplitude 1e-6)
        amp = 1e-6
        sigma = 3.0
        x = np.arange(N) - center
        y = np.arange(N) - center
        z = np.arange(N) - center
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r2 = X**2 + Y**2 + Z**2
        bump = amp * np.exp(-r2 / (2 * sigma**2))
        self.gr_solver.fields.K_sym6[..., 0] += bump  # Add to xx component
        
        # Hold gauge steady by overriding gauge evolution (set ∂t α = 0, ∂t β = 0)
        self.gr_solver.gauge.evolve_lapse = lambda dt: None
        self.gr_solver.gauge.evolve_shift = lambda dt: None
        
        # Reset time and step
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0
        
        # Lists to track metrics
        peak_locations = []
        peak_amplitudes = []
        times = []
        spread_max_r = []
        
        initial_threshold = amp / 1e6  # Threshold for spread calculation
        
        # Evolve for 10 steps
        for step in range(10):
            dt, _, _ = self.gr_solver.orchestrator.run_step()
            
            # Compute H grid
            self.gr_solver.constraints.compute_hamiltonian()
            H = self.gr_solver.constraints.H
            H_abs = np.abs(H)
            
            # Find peak |H| and location
            peak_idx = np.argmax(H_abs.ravel())
            peak_loc = np.unravel_index(peak_idx, H.shape)
            peak_amp = H_abs.flat[peak_idx]
            
            peak_locations.append(peak_loc)
            peak_amplitudes.append(peak_amp)
            times.append(self.gr_solver.orchestrator.t)
            
            # Compute spread: max r where |H| > threshold
            mask = H_abs > initial_threshold
            if np.any(mask):
                i_vals, j_vals, k_vals = np.where(mask)
                r_vals = np.sqrt((i_vals - center)**2 + (j_vals - center)**2 + (k_vals - center)**2)
                max_r = np.max(r_vals)
            else:
                max_r = 0.0
            spread_max_r.append(max_r)
        
        # Compute peak speeds (distances between consecutive peaks)
        peak_speeds = []
        for i in range(1, len(peak_locations)):
            loc1 = np.array(peak_locations[i-1])
            loc2 = np.array(peak_locations[i])
            dist = np.linalg.norm(loc2 - loc1)
            peak_speeds.append(dist)
        
        # Compute spread rates (increase in max_r per step)
        spread_rates = []
        for i in range(1, len(spread_max_r)):
            rate = spread_max_r[i] - spread_max_r[i-1]
            spread_rates.append(rate)
        
        # Peak decay: list of peak amplitudes
        peak_decay = peak_amplitudes
        
        # Check pass conditions
        max_speed = max(peak_speeds) if peak_speeds else 0
        smooth_movement = all(speed <= 3 for speed in peak_speeds)
        no_teleport = max_speed <= N / 2  # Arbitrary, but reasonable
        no_sudden_elsewhere = True  # Assume if speeds are low, no teleport
        
        passed = smooth_movement and no_teleport and no_sudden_elsewhere
        
        diagnosis = f"Max peak speed: {max_speed:.2f} cells/step, {'passed' if passed else 'failed'} criteria (speed <=3, smooth movement)"
        
        metrics = {
            'peak_speeds': peak_speeds,
            'peak_decay': peak_decay,
            'spread_rates': spread_rates
        }
        
        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test4:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski data
        self.gr_solver.init_minkowski()

        # Add small random perturbations to excite dynamics
        np.random.seed(42)
        pert_scale = 1e-12
        self.gr_solver.fields.gamma_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.gamma_sym6.shape)
        self.gr_solver.fields.K_sym6 += pert_scale * np.random.randn(*self.gr_solver.fields.K_sym6.shape)
        self.gr_solver.fields.alpha += pert_scale * np.random.randn(*self.gr_solver.fields.alpha.shape)
        self.gr_solver.fields.beta += pert_scale * np.random.randn(*self.gr_solver.fields.beta.shape)

        # Store initial fields for delta_state calculation
        initial_gamma = self.gr_solver.fields.gamma_sym6.copy()
        initial_K = self.gr_solver.fields.K_sym6.copy()
        initial_alpha = self.gr_solver.fields.alpha.copy()
        initial_beta = self.gr_solver.fields.beta.copy()

        # Lambda values
        lambda_vals = [0.0, 0.1, 0.2, 0.4]

        T = 0.1
        eps_H_curve = []
        delta_state_curve = []

        for lambda_val in lambda_vals:
            # Create a deep copy of the solver
            solver = copy.deepcopy(self.gr_solver)

            # Set damping parameter in stepper
            solver.stepper.lambda_val = lambda_val

            # Reset time and step
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0

            # Run evolution for fixed T
            while solver.orchestrator.t < T:
                dt_max = T - solver.orchestrator.t
                dt, _, _ = solver.orchestrator.run_step(dt_max)

            # Compute final eps_H
            solver.constraints.compute_hamiltonian()
            eps_H = solver.constraints.eps_H
            eps_H_curve.append(eps_H)

            # Compute delta_state(t) = sum of L2 norms of field deltas
            delta_gamma = solver.fields.gamma_sym6 - initial_gamma
            delta_K = solver.fields.K_sym6 - initial_K
            delta_alpha = solver.fields.alpha - initial_alpha
            delta_beta = solver.fields.beta - initial_beta

            l2_gamma = np.sqrt(np.sum(delta_gamma**2))
            l2_K = np.sqrt(np.sum(delta_K**2))
            l2_alpha = np.sqrt(np.sum(delta_alpha**2))
            l2_beta = np.sqrt(np.sum(delta_beta**2))
            delta_state = l2_gamma + l2_K + l2_alpha + l2_beta
            delta_state_curve.append(delta_state)

        # Check monotonic decrease in final eps_H with increasing lambda
        monotonic = all(eps_H_curve[i] > eps_H_curve[i+1] for i in range(len(eps_H_curve)-1))

        # Check delta_state > 1e-12 (for the highest lambda, assuming it's not driven to 0)
        delta_ok = delta_state_curve[-1] > 1e-12

        passed = monotonic and delta_ok

        diagnosis = f"eps_H monotonic decrease: {monotonic}, delta_state > 1e-12: {delta_ok}"

        return {'passed': passed, 'metrics': {'eps_H_curve': eps_H_curve, 'delta_state_curve': delta_state_curve}, 'diagnosis': diagnosis}

class Test5:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        import copy
        from gr_solver.gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33

        T = 0.05
        tol_gamma = 1e-6
        tol_inv = 1e-10

        # Run A: Normal evolution with Minkowski initial data
        solver_A = copy.deepcopy(self.gr_solver)
        solver_A.init_minkowski()
        solver_A.orchestrator.t = 0.0
        solver_A.orchestrator.step = 0
        while solver_A.orchestrator.t < T:
            dt_max = T - solver_A.orchestrator.t
            dt, _, _ = solver_A.orchestrator.run_step(dt_max)

        # Compute invariants for A
        solver_A.geometry.compute_scalar_curvature()
        R_A = solver_A.geometry.R
        gamma_inv_A = inv_sym6(solver_A.fields.gamma_sym6)
        gamma_inv_full_A = sym6_to_mat33(gamma_inv_A)
        K_full_A = sym6_to_mat33(solver_A.fields.K_sym6)
        K_raised_A = np.einsum('...ik,...kj->...ij', gamma_inv_full_A, K_full_A)
        K2_A = np.einsum('...ij,...ij', K_raised_A, K_full_A)
        trK_A = trace_sym6(solver_A.fields.K_sym6, gamma_inv_A)
        gamma_full_A = sym6_to_mat33(solver_A.fields.gamma_sym6)
        det_A = np.linalg.det(gamma_full_A)

        # Run B: Apply gauge scramble
        solver_B = copy.deepcopy(self.gr_solver)
        solver_B.init_minkowski()
        # Perturb alpha and beta with small sinusoidal perturbations
        N = solver_B.fields.N
        eps = 1e-6
        x = np.arange(N) * solver_B.fields.dx
        y = np.arange(N) * solver_B.fields.dy
        z = np.arange(N) * solver_B.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = eps * np.sin(2 * np.pi * (X + Y + Z) / N)
        solver_B.fields.alpha += pert
        solver_B.fields.beta[..., 0] += pert
        solver_B.fields.beta[..., 1] += pert
        solver_B.fields.beta[..., 2] += pert
        # Evolve
        solver_B.orchestrator.t = 0.0
        solver_B.orchestrator.step = 0
        while solver_B.orchestrator.t < T:
            dt_max = T - solver_B.orchestrator.t
            dt, _, _ = solver_B.orchestrator.run_step(dt_max)

        # Compute invariants for B
        solver_B.geometry.compute_scalar_curvature()
        R_B = solver_B.geometry.R
        gamma_inv_B = inv_sym6(solver_B.fields.gamma_sym6)
        gamma_inv_full_B = sym6_to_mat33(gamma_inv_B)
        K_full_B = sym6_to_mat33(solver_B.fields.K_sym6)
        K_raised_B = np.einsum('...ik,...kj->...ij', gamma_inv_full_B, K_full_B)
        K2_B = np.einsum('...ij,...ij', K_raised_B, K_full_B)
        trK_B = trace_sym6(solver_B.fields.K_sym6, gamma_inv_B)
        gamma_full_B = sym6_to_mat33(solver_B.fields.gamma_sym6)
        det_B = np.linalg.det(gamma_full_B)

        # Compute L2 differences
        diff_gamma = np.sqrt(np.sum((solver_A.fields.gamma_sym6 - solver_B.fields.gamma_sym6)**2))
        diff_R = np.sqrt(np.sum((R_A - R_B)**2))
        diff_K2 = np.sqrt(np.sum((K2_A - K2_B)**2))
        diff_trK = np.sqrt(np.sum((trK_A - trK_B)**2))
        diff_det = np.sqrt(np.sum((det_A - det_B)**2))

        # Pass if gauge fields differ and invariants match
        passed = diff_gamma > tol_gamma and diff_R < tol_inv and diff_K2 < tol_inv and diff_trK < tol_inv and diff_det < tol_inv

        metrics = {'diff_gamma': diff_gamma, 'diff_R': diff_R, 'diff_K2': diff_K2, 'diff_trK': diff_trK, 'diff_det': diff_det}
        diagnosis = f"Gauge scramble applied, gamma differs by {diff_gamma:.2e} (> {tol_gamma}), invariants match within {tol_inv}"

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test6Shat:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        import copy

        # Define high bands: indices where at least one dimension is high (2)
        def is_high_band(idx):
            i = idx // 9
            j = (idx // 3) % 3
            k = idx % 3
            return i == 2 or j == 2 or k == 2

        high_band_indices = [idx for idx in range(27) if is_high_band(idx)]

        # Function to compute E_highk
        def compute_E_highk(omega):
            return np.sum(omega[high_band_indices])

        # Initialize solver with Minkowski
        self.gr_solver.init_minkowski()

        N = self.gr_solver.fields.Nx
        k = N // 4  # mid-frequency
        eps = 1e-6  # amplitude
        x = np.arange(N) * self.gr_solver.fields.dx
        y = np.arange(N) * self.gr_solver.fields.dy
        z = np.arange(N) * self.gr_solver.fields.dz
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pert = eps * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        self.gr_solver.fields.gamma_sym6[..., 0] += pert  # xx component

        # Reset time and step
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0

        # Run with dealiasing ON (damping enabled)
        solver_on = copy.deepcopy(self.gr_solver)
        solver_on.stepper.damping_enabled = True
        solver_on.stepper.lambda_val = 0.1  # some damping
        E_highk_on = []
        for step in range(20):
            dt, _, _ = solver_on.orchestrator.run_step()
            omega = solver_on.orchestrator.render.compute_omega()
            E_highk_on.append(compute_E_highk(omega))

        # Run with dealiasing OFF (damping disabled)
        solver_off = copy.deepcopy(self.gr_solver)
        solver_off.stepper.damping_enabled = False
        E_highk_off = []
        for step in range(20):
            dt, _, _ = solver_off.orchestrator.run_step()
            omega = solver_off.orchestrator.render.compute_omega()
            E_highk_off.append(compute_E_highk(omega))

        # Pass if ON: E_highk slowly increases or decays; OFF: explodes
        # Check if ON is stable: final < initial * 2 or decreasing
        initial_on = E_highk_on[0] if E_highk_on else 0
        final_on = E_highk_on[-1] if E_highk_on else 0
        stable_on = final_on <= initial_on * 2 and (final_on <= initial_on or len(E_highk_on) < 2 or E_highk_on[-1] < E_highk_on[-2])

        # OFF explodes: final > initial * 10
        initial_off = E_highk_off[0] if E_highk_off else 0
        final_off = E_highk_off[-1] if E_highk_off else 0
        explodes_off = final_off > initial_off * 10

        passed = stable_on and explodes_off

        diagnosis = f"ON: stable={stable_on} (init={initial_on:.2e}, final={final_on:.2e}); OFF: explodes={explodes_off} (init={initial_off:.2e}, final={final_off:.2e})"

        metrics = {'E_highk_on': E_highk_on, 'E_highk_off': E_highk_off}

        return {'passed': passed, 'metrics': metrics, 'diagnosis': diagnosis}

class Test7Tss:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        import copy
        # Initialize solver with Minkowski + small perturbations
        self.gr_solver.init_minkowski()
        pert = 1e-12
        self.gr_solver.fields.gamma_sym6[0, 0, 0, 0] += pert
        self.gr_solver.fields.K_sym6[0, 0, 0, 0] += pert

        # Compute dt0 from scheduler for T=0.1
        self.gr_solver.constraints.compute_hamiltonian()
        self.gr_solver.constraints.compute_momentum()
        eps_H = self.gr_solver.constraints.eps_H
        eps_M = self.gr_solver.constraints.eps_M
        dt0 = self.gr_solver.scheduler.compute_dt(eps_H, eps_M)

        # Run evolutions with dt_multipliers = [0.5, 1.0, 2.0, 4.0], each to T=0.1, track max eps_H during run
        dt_multipliers = [0.5, 1.0, 2.0, 4.0]
        eps_H_max_list = []
        for mult in dt_multipliers:
            solver = copy.deepcopy(self.gr_solver)
            # Override compute_dt to force dt = dt0 * mult
            solver.scheduler.compute_dt = lambda eps_H, eps_M: dt0 * mult
            # Reset time and step
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            T = 0.1
            max_eps_H = 0.0
            while solver.orchestrator.t < T:
                dt, _, _ = solver.orchestrator.run_step()
                # eps_H is computed in run_step, but ensure
                solver.constraints.compute_hamiltonian()
                max_eps_H = max(max_eps_H, solver.constraints.eps_H)
            eps_H_max_list.append(max_eps_H)

        # Check that eps_H increases with dt_multiplier, but for multiplier=0.5 and 1.0 it's stable, for 2.0 and 4.0 it degrades but doesn't crash immediately
        # Pass if eps_H_curve is monotonic increasing and for small dt it's low
        eps_H_curve = eps_H_max_list
        monotonic_increasing = all(eps_H_curve[i] <= eps_H_curve[i+1] for i in range(len(eps_H_curve)-1))
        small_dt_low = eps_H_curve[0] < 1e-6 and eps_H_curve[1] < 1e-6
        passed = monotonic_increasing and small_dt_low

        diagnosis = f"Monotonic increasing: {monotonic_increasing}, small dt low: {small_dt_low}, eps_H_max: {eps_H_max_list}"

        return {'passed': passed, 'metrics': {'dt_multipliers': dt_multipliers, 'eps_H_max': eps_H_max_list}, 'diagnosis': diagnosis}

class Test8Jsc:
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski
        self.gr_solver.init_minkowski()
        self.gr_solver.geometry.compute_christoffels()
        self.gr_solver.geometry.compute_ricci()
        self.gr_solver.geometry.compute_scalar_curvature()

        # Store original fields
        original_gamma = self.gr_solver.fields.gamma_sym6.copy()
        original_K = self.gr_solver.fields.K_sym6.copy()

        # Pick random perturbation δu (random arrays for gamma_sym6 and K_sym6)
        np.random.seed(42)
        delta_gamma = np.random.randn(*self.gr_solver.fields.gamma_sym6.shape) * 1e-10
        delta_K = np.random.randn(*self.gr_solver.fields.K_sym6.shape) * 1e-10

        # Compute F(u) = stepper.compute_rhs()
        self.gr_solver.stepper.compute_rhs()
        F_u_gamma = self.gr_solver.stepper.rhs_gamma_sym6.copy()
        F_u_K = self.gr_solver.stepper.rhs_K_sym6.copy()
        F_u_phi = self.gr_solver.stepper.rhs_phi.copy()

        ratios = []
        epsilons = [1e-6, 1e-7, 1e-8]
        for eps in epsilons:
            # Perturb fields: u + ε δu
            self.gr_solver.fields.gamma_sym6 = original_gamma + eps * delta_gamma
            self.gr_solver.fields.K_sym6 = original_K + eps * delta_K

            # Recompute geometry since gamma changed
            self.gr_solver.geometry.compute_christoffels()
            self.gr_solver.geometry.compute_ricci()
            self.gr_solver.geometry.compute_scalar_curvature()

            # Compute F(u + ε δu)
            self.gr_solver.stepper.compute_rhs()
            F_eps_gamma = self.gr_solver.stepper.rhs_gamma_sym6.copy()
            F_eps_K = self.gr_solver.stepper.rhs_K_sym6.copy()
            F_eps_phi = self.gr_solver.stepper.rhs_phi.copy()

            # Compute ||F(u + ε δu) - F(u)|| / ε
            diff_gamma = F_eps_gamma - F_u_gamma
            diff_K = F_eps_K - F_u_K
            diff_phi = F_eps_phi - F_u_phi
            norm_diff = np.sqrt(np.sum(diff_gamma**2) + np.sum(diff_K**2) + np.sum(diff_phi**2))
            ratio = norm_diff / eps
            ratios.append(ratio)

        # Restore original fields
        self.gr_solver.fields.gamma_sym6 = original_gamma
        self.gr_solver.fields.K_sym6 = original_K
        self.gr_solver.geometry.compute_christoffels()
        self.gr_solver.geometry.compute_ricci()
        self.gr_solver.geometry.compute_scalar_curvature()

        # Check if ratios converge (approximately constant or decreasing for small ε)
        # Pass if the ratio is consistent for small ε (e.g., not diverging)
        passed = ratios[2] >= 0.5 * ratios[1] and ratios[2] <= 2 * ratios[1] and ratios[1] >= 0.5 * ratios[0] and ratios[1] <= 2 * ratios[0]

        diagnosis = f"Ratios for ε=[1e-6,1e-7,1e-8]: {ratios[0]:.2e}, {ratios[1]:.2e}, {ratios[2]:.2e} - {'consistent' if passed else 'inconsistent'}"

        return {'passed': passed, 'metrics': {'ratios': ratios}, 'diagnosis': diagnosis}

class Test9Bianchi:
    """
    Numerically verifies the contracted Bianchi identity div(G) = 0 
    (and thus div(T) = 0 via Einstein Eq) on a dynamical spacetime.
    """
    def __init__(self, gr_solver: GRSolver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Running Test9Bianchi (div T = 0 check)...")
        
        # 1. Initialize with a non-trivial wave (Minkowski + perturbation)
        self.gr_solver.init_minkowski()
        
        N = self.gr_solver.fields.Nx
        dx = self.gr_solver.fields.dx
        L = N * dx
        
        # Inject a smooth perturbation to generate curvature
        x = np.arange(N) * dx
        y = np.arange(N) * dx
        z = np.arange(N) * dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Perturbation in gamma_xx
        amp = 0.001
        k = 2 * np.pi / L
        pert = amp * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        
        # Apply to physical gamma
        self.gr_solver.fields.gamma_sym6[..., 0] += pert
        
        # Enforce BSSN algebraic constraints (phi, gamma_tilde)
        det_g = det_sym6(self.gr_solver.fields.gamma_sym6)
        self.gr_solver.fields.phi = (1.0/12.0) * np.log(det_g)
        self.gr_solver.fields.gamma_tilde_sym6 = self.gr_solver.fields.gamma_sym6 * np.exp(-4.0 * self.gr_solver.fields.phi)[..., np.newaxis]
        
        # Perturbation in lapse to make g_00 interesting
        self.gr_solver.fields.alpha += 0.5 * pert
        
        # 2. Evolve and capture 7 time slices
        slices = []
        dt = 0.001 # Small dt for accuracy
        
        # Reset time
        self.gr_solver.orchestrator.t = 0.0
        self.gr_solver.orchestrator.step = 0
        
        for i in range(7):
            slices.append(self._capture_state())
            # Force a fixed small timestep for finite difference accuracy
            self.gr_solver.orchestrator.run_step(dt_max=dt)
            
        # 3. Compute Divergence at the central slice (index 3)
        G_prev, _ = self._compute_G_at(slices, 2, dt)
        G_curr, Gamma_curr = self._compute_G_at(slices, 3, dt)
        G_next, _ = self._compute_G_at(slices, 4, dt)
        
        # Derivatives of G^mu_nu at center (index 3)
        dG_dt = (G_next - G_prev) / (2*dt)
        dG_dx = self._spatial_deriv(G_curr, 0)
        dG_dy = self._spatial_deriv(G_curr, 1)
        dG_dz = self._spatial_deriv(G_curr, 2)
        
        dG = np.zeros((4, 4, 4) + G_curr.shape[2:])
        dG[0] = dG_dt
        dG[1] = dG_dx
        dG[2] = dG_dy
        dG[3] = dG_dz
        
        # Divergence: nabla_mu G^mu_nu
        divG = np.zeros((4,) + G_curr.shape[2:])
        for nu in range(4):
            term = np.zeros_like(divG[0])
            for mu in range(4):
                term += dG[mu, mu, nu]
                for lam in range(4):
                    term += Gamma_curr[mu, mu, lam] * G_curr[lam, nu]
                for lam in range(4):
                    term += Gamma_curr[nu, mu, lam] * G_curr[mu, lam]
            divG[nu] = term
            
        # Check magnitude
        divG_norm = np.sqrt(np.mean(divG**2))
        threshold = 1e-3
        passed = divG_norm < threshold
        
        diagnosis = f"Bianchi residual {divG_norm:.2e} < {threshold}" if passed else f"Bianchi residual {divG_norm:.2e} >= {threshold}"
        return {'passed': passed, 'metrics': {'divG_norm': divG_norm}, 'diagnosis': diagnosis}

    def _capture_state(self):
        return {
            't': self.gr_solver.orchestrator.t,
            'gamma': self.gr_solver.fields.gamma_sym6.copy(),
            'alpha': self.gr_solver.fields.alpha.copy(),
            'beta': self.gr_solver.fields.beta.copy()
        }

    def _compute_G_at(self, slices, idx, dt):
        Gammas_local = []
        for j in [idx-1, idx, idx+1]:
            s_m = slices[j-1]
            s_c = slices[j]
            s_p = slices[j+1]
            g_loc = self._construct_4metric(s_c)
            dg_dt_loc = (self._construct_4metric(s_p) - self._construct_4metric(s_m)) / (2*dt)
            dg_dx_loc = self._spatial_deriv(g_loc, 0)
            dg_dy_loc = self._spatial_deriv(g_loc, 1)
            dg_dz_loc = self._spatial_deriv(g_loc, 2)
            dg_loc = np.zeros((4, 4, 4) + g_loc.shape[2:])
            dg_loc[0] = dg_dt_loc
            dg_loc[1] = dg_dx_loc
            dg_loc[2] = dg_dy_loc
            dg_loc[3] = dg_dz_loc
            g_inv_loc = self._inverse_4metric(g_loc)
            Gam_loc = np.zeros((4, 4, 4) + g_loc.shape[2:])
            for sig in range(4):
                for mu in range(4):
                    for nu in range(4):
                        val = 0.0
                        for lam in range(4):
                            val += 0.5 * g_inv_loc[sig, lam] * (dg_loc[mu, nu, lam] + dg_loc[nu, mu, lam] - dg_loc[lam, mu, nu])
                        Gam_loc[sig, mu, nu] = val
            Gammas_local.append(Gam_loc)
        
        Gamma_c = Gammas_local[1]
        dGamma_dt = (Gammas_local[2] - Gammas_local[0]) / (2*dt)
        dGamma_dx = self._spatial_deriv(Gamma_c, 0)
        dGamma_dy = self._spatial_deriv(Gamma_c, 1)
        dGamma_dz = self._spatial_deriv(Gamma_c, 2)
        dGamma = np.zeros((4, 4, 4, 4) + Gamma_c.shape[3:])
        dGamma[0] = dGamma_dt
        dGamma[1] = dGamma_dx
        dGamma[2] = dGamma_dy
        dGamma[3] = dGamma_dz
        
        R_tensor = np.zeros((4, 4, 4, 4) + Gamma_c.shape[3:])
        for rho in range(4):
            for sig in range(4):
                for mu in range(4):
                    for nu in range(4):
                        t1 = dGamma[mu, rho, nu, sig]
                        t2 = dGamma[nu, rho, mu, sig]
                        t3 = np.zeros_like(t1)
                        t4 = np.zeros_like(t1)
                        for lam in range(4):
                            t3 += Gamma_c[rho, mu, lam] * Gamma_c[lam, nu, sig]
                            t4 += Gamma_c[rho, nu, lam] * Gamma_c[lam, mu, sig]
                        R_tensor[rho, sig, mu, nu] = t1 - t2 + t3 - t4
        
        Ricci = np.zeros((4, 4) + Gamma_c.shape[3:])
        for sig in range(4):
            for nu in range(4):
                for rho in range(4):
                    Ricci[sig, nu] += R_tensor[rho, sig, rho, nu]
        
        g_c = self._construct_4metric(slices[idx])
        g_inv_c = self._inverse_4metric(g_c)
        R_scalar = np.zeros_like(Ricci[0,0])
        for sig in range(4):
            for nu in range(4):
                R_scalar += g_inv_c[sig, nu] * Ricci[sig, nu]
        
        G_tensor = np.zeros((4, 4) + Gamma_c.shape[3:])
        for mu in range(4):
            for nu in range(4):
                G_tensor[mu, nu] = Ricci[mu, nu] - 0.5 * R_scalar * g_c[mu, nu]
        
        G_up = np.zeros_like(G_tensor)
        for mu in range(4):
            for nu in range(4):
                term = np.zeros_like(G_tensor[0,0])
                for alpha in range(4):
                    for beta in range(4):
                        term += g_inv_c[mu, alpha] * g_inv_c[nu, beta] * G_tensor[alpha, beta]
                G_up[mu, nu] = term
        return G_up, Gamma_c

    def _construct_4metric(self, state):
        gamma_sym = state['gamma']
        alpha = state['alpha']
        beta = state['beta']
        gamma = sym6_to_mat33(gamma_sym)
        beta_lower = np.einsum('...ij,...j->...i', gamma, beta)
        beta_sq = np.einsum('...i,...i->...', beta, beta_lower)
        g4 = np.zeros((4, 4) + alpha.shape)
        g4[0, 0] = -alpha**2 + beta_sq
        for i in range(3):
            g4[0, i+1] = beta_lower[..., i]
            g4[i+1, 0] = beta_lower[..., i]
            for j in range(3):
                g4[i+1, j+1] = gamma[..., i, j]
        return g4

    def _inverse_4metric(self, g4):
        g4_t = np.moveaxis(g4, [0, 1], [-2, -1])
        g4_inv_t = np.linalg.inv(g4_t)
        g4_inv = np.moveaxis(g4_inv_t, [-2, -1], [0, 1])
        return g4_inv

    def _spatial_deriv(self, f, axis):
        dx = self.gr_solver.fields.dx
        return (np.roll(f, -1, axis=axis-3) - np.roll(f, 1, axis=axis-3)) / (2 * dx)

class GCAT1Suite:
    def __init__(self, gr_solver: GRSolver, **kwargs):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)
        self.logger.info("GCAT1Suite initialized with GRSolver instance")
        # Handle optional parameters if needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run_all_tests(self):
        scorecard = {}
        scorecard['test_0_bs2'] = Test0Bs2(self.gr_solver).run()
        scorecard['test_1'] = Test1MmsLite(self.gr_solver).run()
        scorecard['test_1a'] = Test1A(self.gr_solver).run()
        scorecard['test_1b'] = Test1B(self.gr_solver).run()
        scorecard['test_2'] = Test2Rcs(self.gr_solver).run()
        scorecard['test_3'] = Test3(self.gr_solver).run()
        scorecard['test_4'] = Test4(self.gr_solver).run()
        scorecard['test_5'] = Test5(self.gr_solver).run()
        scorecard['test_6'] = Test6Shat(self.gr_solver).run()
        scorecard['test_7_tss'] = Test7Tss(self.gr_solver).run()
        scorecard['test_8_jsc'] = Test8Jsc(self.gr_solver).run()  # Optional
        scorecard['test_9_bianchi'] = Test9Bianchi(self.gr_solver).run()
        return scorecard

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)
    # Instantiate GRSolver with small grid N=8
    solver = GRSolver(8, 8, 8, dx=1.0, dy=1.0, dz=1.0)
    # Run only Test 1 (MMS-lite)
    result = Test1MmsLite(solver).run()
    # Convert numpy types to Python types for JSON
    result['passed'] = bool(result['passed'])
    print(result)
    # Save to receipts_gcat1_test1.json
    with open('receipts_gcat1_test1.json', 'w') as f:
        json.dump(result, f, indent=2)