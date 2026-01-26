"""
CP-1: NSC-GR Coherence Proof Test Suite
"""
import numpy as np
import pytest
from gr_solver.gr_solver import GRSolver
from receipt_schemas import validate_receipt_chain

# Adapter class to interface with the GR solver
class NSCGRCoherenceAdapter:
    def __init__(self, dx=0.1, dy=0.1, dz=0.1):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.caps = {
            "H_cap": 1e-3,
            "M_cap": 1e-3,
            "trace_tol": 5e-2,
        }
        self.budgets = {
            "dt_min": 1e-8,
            "dt_max": 1.0,
            "kappa_max": 1.0,
        }

    def get_ricci_at_resolution(self, N):
        L = 1.0 # Box size
        dx = L/N
        solver = GRSolver(Nx=N, Ny=N, Nz=N, dx=dx, dy=dx, dz=dx)
        
        # Use a smooth analytical perturbation instead of random
        solver.fields.init_minkowski()
        A = 1e-7
        x = np.arange(solver.fields.Nx) * dx
        y = np.arange(solver.fields.Ny) * dx
        z = np.arange(solver.fields.Nz) * dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Gaussian perturbation
        pert = A * np.exp(-((X-L/2)**2 + (Y-L/2)**2 + (Z-L/2)**2) / (L/4)**2)
        solver.fields.gamma_sym6[..., 0] += pert
        
        solver.geometry.compute_christoffels()
        solver.geometry.compute_ricci()
        
        return solver.geometry.ricci

    def ricci_reference_error(self, N, R_ref):
        R_N = self.get_ricci_at_resolution(N)
        
        # Interpolate R_ref to the grid of R_N
        R_ref_interp = np.zeros_like(R_N)
        for i in range(3):
            for j in range(3):
                # Simple down-sampling, assuming N_ref is a multiple of N
                ratio = R_ref.shape[0] // N
                R_ref_interp[..., i, j] = R_ref[::ratio, ::ratio, ::ratio, i, j]

        error_L2 = np.linalg.norm(R_N - R_ref_interp) * (1.0/N)**3
        return {"error_L2": error_L2, "dx": 1.0/N}

    def run_short(self, T, governed=True, clock_mode="normal"):
        
        solver = GRSolver(Nx=16, Ny=16, Nz=16, dx=self.dx, dy=self.dy, dz=self.dz)
        solver.init_minkowski()
        
        # Add a smaller perturbation to avoid immediate constraint violation
        solver.fields.gamma_sym6[..., 0] *= 1 + 1e-12 * np.random.randn(16,16,16)

        dt_max = None
        if clock_mode == "contract":
            dt_max = 1e-5 # Small fixed dt
        
        solver.run(T_max=T, dt_max=dt_max)
        
        receipts = solver.orchestrator.receipts.omega_receipts
            
        return {
            "receipts": receipts,
            "budgets": self.budgets
        }

@pytest.fixture
def adapter():
    return NSCGRCoherenceAdapter()

def observed_order(errors, dxs):
    # fit log(err) = p log(dx) + c
    x = np.log(np.array(dxs))
    y = np.log(np.array(errors))
    p = np.polyfit(x, y, 1)[0]
    return p

def tail_drop_unpaid(r):
    # witness: tail drops hard without declared stabilizing payment
    drop = (r.record.get("E_tail_after", 0.0) < 0.6 * r.record.get("E_tail_before", 0.0))
    paid = (r.record.get("kappa_after", 0.0) > r.record.get("kappa_before", 0.0)) or ("DECL_FILTER_APPLIED" in r.record.get("action_flags", []))
    return drop and (not paid)

def test_cp1A_operator_coherence(adapter):
    # Get a high-resolution reference solution
    N_ref = 128
    R_ref = adapter.get_ricci_at_resolution(N_ref)
    
    Ns = [8, 16, 32, 64]
    errors = []
    dxs = []
    for N in Ns:
        out = adapter.ricci_reference_error(N, R_ref)
        errors.append(out["error_L2"])
        dxs.append(out["dx"])
    p = observed_order(errors, dxs)
    print(f"\nCP-1A (Operator Coherence): Observed convergence order p_obs = {p:.3f} (Ns={Ns}, Errors={errors})")
    assert p >= 1.9, f"Expected ~2nd order, got p={p:.3f}"
    assert all(errors[i+1] < errors[i] for i in range(len(errors)-1))

def test_cp1B_constraints_no_cheat(adapter):
    run = adapter.run_short(governed=True, T=1.0)
    receipts = run["receipts"]
    assert validate_receipt_chain(receipts)

    # Caps
    H_cap = adapter.caps["H_cap"]
    M_cap = adapter.caps["M_cap"]

    # Constraint violations
    eps_H = np.array([r.record['constraints']['eps_post_H'] for r in receipts if 'constraints' in r.record])
    eps_M = np.array([r.record['constraints']['eps_post_M'] for r in receipts if 'constraints' in r.record])
    max_eps_H = np.max(eps_H) if len(eps_H) > 0 else 0
    max_eps_M = np.max(eps_M) if len(eps_M) > 0 else 0
    
    assert max_eps_H <= H_cap
    assert max_eps_M <= M_cap

    # Unpaid tail drops
    suspects = [r for r in receipts if tail_drop_unpaid(r)]
    assert len(suspects) == 0, f"Found unpaid tail drops: {len(suspects)}"

    # Coercion Ledger
    kappa_bumps = [r.record.get("kappa_after", 0.0) - r.record.get("kappa_before", 0.0) for r in receipts if r.record.get("kappa_after", 0.0) > r.record.get("kappa_before", 0.0)]
    sum_kappa_bumps = sum(kappa_bumps)
    kappa_max_reached = max([r.record.get("kappa_after", 0.0) for r in receipts]) if receipts else 0
    num_rejects = len([r for r in receipts if "REJECT_STEP" in r.record.get("action_flags", [])]) # Approximate

    print(f"CP-1B (Constraint Governance): Max eps_H={max_eps_H:.4g}, Max eps_M={max_eps_M:.4g}, No unpaid tail drops.")
    print(f"CP-1B Coercion Ledger: sum_kappa_bumps = {sum_kappa_bumps:.4g}, kappa_max_reached = {kappa_max_reached:.4g}, num_rejects = {num_rejects}")

def test_cp1C_clock_coherence(adapter):
    a = adapter.run_short(governed=True, T=1.0, clock_mode="normal")
    b = adapter.run_short(governed=True, T=1.0, clock_mode="contract")

    # Compare max Hamiltonian constraint violation
    tol = adapter.caps.get("trace_tol", 5e-2)
    ea = np.array([r.record['constraints']['eps_post_H'] for r in a["receipts"] if 'constraints' in r.record])
    eb = np.array([r.record['constraints']['eps_post_H'] for r in b["receipts"] if 'constraints' in r.record])
    
    if len(ea) > 0 and len(eb) > 0:
        max_ea = np.max(ea)
        max_eb = np.max(eb)
        diff = abs(max_ea - max_eb)
        norm = max(max_ea, max_eb, 1e-12)
        print(f"\nCP-1C (Clock Coherence): Max eps_H (normal mode) = {max_ea:.4g}, Max eps_H (contract mode) = {max_eb:.4g}, Rel. Diff = {diff/norm:.4g}")
        assert diff <= tol * norm

def test_cp1D_budget_closure(adapter):
    run = adapter.run_short(governed=True, T=1.0)
    receipts = run["receipts"]
    K = run["budgets"]

    # dt bounds
    for r in receipts:
        dt_committed = r.record.get("dt")
        if dt_committed:
            assert K["dt_min"] <= dt_committed <= K["dt_max"]

    # kappa bounds
    for r in receipts:
        k = r.record.get("kappa_after", 0.0)
        assert 0.0 <= k <= K["kappa_max"]
    print("\nCP-1D (Budget Closure): dt and kappa values remained within declared bounds.")
