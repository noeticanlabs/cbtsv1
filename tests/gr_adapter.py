"""
GR solver adapter for test suite.
"""

import numpy as np
import copy
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List
from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import inv_sym6, sym6_to_mat33, mat33_to_sym6, det_sym6

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
    # signature: exact_state_arrays(N, L, t) -> dict[str, array]
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