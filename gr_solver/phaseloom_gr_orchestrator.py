# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "PhaseLoom"
]
LEXICON_SYMBOLS = {
    "\\Delta t": "CTL_time.step",
    "\\Psi": "UFE_state",
    "T_phys": "PhaseLoom.thread_phys",
    "T_curv": "PhaseLoom.thread_curv",
    "T_cons": "PhaseLoom.thread_cons",
    "T_gauge": "PhaseLoom.thread_gauge",
    "T_diff": "PhaseLoom.thread_diff",
    "T_io": "PhaseLoom.thread_io"
}

import numpy as np
import copy
import logging
from aeonic_clocks import AeonicClockPack
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_receipts import AeonicReceipts
from .logging_config import Timer, array_stats
from .phaseloom_threads_gr import GRPhaseLoomThreads, compute_omega_current
from .phaseloom_rails_gr import GRPhaseLoomRails
from .phaseloom_receipts_gr import GRPhaseLoomReceipts
from .phaseloom_render_gr import GRPhaseLoomRender
from .phaseloom_gr_adapter import GRPhaseLoomAdapter
from .phaseloom_octaves import PhaseLoomOctaves
from .phaseloom_gr_controller import GRPhaseLoomController
from .gr_scheduler import GRScheduler
from orchestrator_contract_memory import OrchestratorContractWithMemory

logger = logging.getLogger('gr_solver.orchestrator')

class PhaseLoomMemory:
    def __init__(self, fields):
        self.fields = fields
        self.step_count = 0
        self.last_loom_step = 0
        self.prev_K = None
        self.prev_gamma = None
        self.prev_residual_slope = None
        self.summary = {}
        self.D_max_prev = 0.0
        self.tainted = False

    def should_compute_loom(self, step, K, gamma, residual_slope, rollback_occurred):
        self.step_count = step
        delta_K = np.max(np.abs(K - self.prev_K)) if self.prev_K is not None else 0.0
        delta_gamma = np.max(np.abs(gamma - self.prev_gamma)) if self.prev_gamma is not None else 0.0
        if step - self.last_loom_step >= 4:
            # Cheap proxy gate: skip FFT if changes are tiny
            if delta_K < 1e-4 and delta_gamma < 1e-4 and abs(residual_slope) < 1e-5:
                return False
            return True
        if delta_K > 1e-3:
            return True
        if delta_gamma > 1e-3:
            return True
        if abs(residual_slope) > 1e-5:
            return True
        if rollback_occurred:
            return True
        return False

    def post_loom_update(self, loom_data):
        self.summary = loom_data
        self.D_max_prev = loom_data.get('D_max', 0.0)
        self.last_loom_step = self.step_count

    def honesty_check(self, skipped):
        if skipped and self.D_max_prev > 1e-2:
            return True
        return False

class GRPhaseLoomOrchestrator:
    def __init__(self, fields, geometry, constraints, gauge, stepper, ledger, memory_contract=None, phaseloom=None, eps_H_target=1e-10, eps_M_target=1e-10, m_det_min=0.2, aeonic_mode=True):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.stepper = stepper
        self.ledger = ledger
        self.aeonic_mode = aeonic_mode

        self.threads = GRPhaseLoomThreads(fields, eps_H_target, eps_M_target, m_det_min)
        self.rails = GRPhaseLoomRails(fields)
        self.receipts = GRPhaseLoomReceipts()
        self.render = GRPhaseLoomRender(fields, geometry, constraints)

        # Aeonic PhaseLoom components
        self.adapter = GRPhaseLoomAdapter(fields.Nx, fields.Ny, fields.Nz)
        self.octaves = PhaseLoomOctaves()
        self.controller = GRPhaseLoomController(O_max=self.octaves.O_max)
        self.scheduler = GRScheduler(fields)

        # Aeonic Memory System
        if memory_contract is None:
            self.clocks = AeonicClockPack()
            self.aeonic_receipts = AeonicReceipts()
            self.memory_bank = AeonicMemoryBank(self.clocks, self.aeonic_receipts)
            self.memory_contract = AeonicMemoryContract(self.memory_bank, self.aeonic_receipts)
        else:
            self.memory_contract = memory_contract
            self.memory_bank = self.memory_contract.memory_bank  # for backward compatibility
            self.clocks = self.memory_bank.clock
            self.aeonic_receipts = self.memory_bank.receipts
        self.memory = self.memory_bank  # alias for backward compatibility
        self.orchestrator_memory = OrchestratorContractWithMemory(self.memory_contract)
        self.loom_memory = PhaseLoomMemory(self.fields)

        # Spectral Cache
        from .spectral.cache import SpectralCache
        self.spectral_cache = SpectralCache(self.fields.Nx, self.fields.Ny, self.fields.Nz, self.fields.dx, self.fields.dy, self.fields.dz)
        # Cache in Tier3
        bytes_est = (self.spectral_cache.kx.nbytes + self.spectral_cache.ky.nbytes + self.spectral_cache.kz.nbytes +
                     self.spectral_cache.k2.nbytes + self.spectral_cache.dealias_mask.nbytes +
                     self.spectral_cache.kx_bin.nbytes + self.spectral_cache.ky_bin.nbytes + self.spectral_cache.kz_bin.nbytes)
        self.memory.put("spectral_cache", 3, self.spectral_cache, bytes_est, ttl_l=1000000, ttl_s=1000000, recompute_cost_est=1000.0, risk_score=0.0, tainted=False, regime_hashes=[])

        self.t = 0.0
        self.step = 0
        self.rollback_count = 0
        self.rollback_reason = None
        self.t_expected = 0.0
        self.tainted = False
        self.disable_suggestions_steps = 0
        self.accepted_step_count = 0
        self.constraint_eval_count_h = 0
        self.constraint_eval_count_m = 0

        # Aeonic counters
        self.attempt_id = 0
        self.step_id = 0
        self.tau = 0.0

        # dt policy
        self.dt_min = 1e-8
        self.dt_shrink = 0.5
        self.dt_grow = 1.2
        self.max_attempts_per_step = 20

        # Memory stores
        self.M_solve = []
        self.M_step = []

        # Policy hash
        self.policy_hash = "policy_v1"

        # Track previous for rates and spectral
        self.eps_H_prev = None
        self.eps_M_prev = None
        self.m_det_prev = None
        self.dt_prev = None
        self.prev_K = None
        self.prev_gamma = None
        self.D_max_prev = 0.0
        self.dt_loom_prev = None
        self.loom_active = False

    def sem_safe_compute_residuals(self):
        """
        Returns (eps_H, eps_M, sem_ok, sem_reason).
        sem_ok=False means: do NOT proceed; reject immediately.
        """
        try:
            self.constraints.compute_hamiltonian()
            self.constraint_eval_count_h += 1
            self.constraints.compute_momentum()
            self.constraint_eval_count_m += 1
            self.constraints.compute_residuals()
            eps_H, eps_M = self.constraints.eps_H, self.constraints.eps_M
        except Exception as e:
            return float("inf"), float("inf"), False, f"SEM:compute_residuals_exception:{type(e).__name__}:{e}"

        # Guard against NaN/Inf
        if not np.isfinite(eps_H) or not np.isfinite(eps_M):
            return float("inf"), float("inf"), False, "SEM:residual_nonfinite"

        return float(eps_H), float(eps_M), True, "ok"

    def make_attempt_receipt(self, *, dt, eps_H, eps_M, sem_ok, sem_reason,
                             gate_reason, rail_margins, dominant_thread,
                             accepted, action, extra=None):
        prev_residual = (self.eps_H_prev + self.eps_M_prev) if self.eps_H_prev is not None else 0.0
        current_residual = eps_H + eps_M
        delta_residual = current_residual - prev_residual
        rel_delta = current_residual / prev_residual if prev_residual > 0 else float('inf')
        r = {
            "kappa": {"o": getattr(self, "orch_id", 0), "s": self.step_id, "mu": None},
            "attempt_id": self.attempt_id,
            "step_id": self.step_id,
            "t": float(self.t),
            "tau": float(self.tau),
            "dt": float(dt),
            "eps_H": float(eps_H),
            "eps_M": float(eps_M),
            "sem_ok": bool(sem_ok),
            "sem_reason": sem_reason,
            "gate_reason": gate_reason,
            "rail_margins": rail_margins,
            "dominant_thread": dominant_thread,
            "accepted": bool(accepted),
            "action": action,
            "policy_hash": self.policy_hash,
            "dominant_clock": dominant_thread,
            "attempt_idx": self.attempt_id,
            "delta_residual": float(delta_residual),
            "rel_delta": float(rel_delta),
        }
        if extra:
            r.update(extra)
        return r

    def make_step_receipt(self, *, dt, eps_H_before, eps_M_before, eps_H_after, eps_M_after,
                          dominant_thread, rail_margins_after, perf=None, extra=None):
        before_residual = eps_H_before + eps_M_before
        after_residual = eps_H_after + eps_M_after
        delta_residual = after_residual - before_residual
        rel_delta = after_residual / before_residual if before_residual > 0 else float('inf')
        r = {
            "kappa": {"o": getattr(self, "orch_id", 0), "s": self.step_id, "mu": None},
            "step_id": self.step_id,
            "t": float(self.t),
            "tau": float(self.tau),
            "dt": float(dt),
            "eps_before": {"H": float(eps_H_before), "M": float(eps_M_before)},
            "eps_after": {"H": float(eps_H_after), "M": float(eps_M_after)},
            "dominant_thread": dominant_thread,
            "rail_margins": rail_margins_after,
            "policy_hash": self.policy_hash,
            "dominant_clock": dominant_thread,
            "attempt_idx": self.attempt_id,
            "delta_residual": float(delta_residual),
            "rel_delta": float(rel_delta),
        }
        if perf:
            r["perf"] = perf
        if extra:
            r.update(extra)
        return r

    def run_step(self, dt_max=None):
        """Execute one PhaseLoom step: Sense, Propose, Decide, Predict, Commit, Verify, Rail-enforce, Receipt, Render"""

        logger.debug("Starting orchestrator run_step", extra={
            "extra_data": {
                "step": self.step,
                "t": self.t,
                "rollback_count": self.rollback_count
            }
        })
        rollback_count_prev = self.rollback_count

        # Detailed profiling timers
        sense_timer = Timer("sense")
        propose_timer = Timer("propose")
        decide_timer = Timer("decide")
        predict_timer = Timer("predict")
        commit_timer = Timer("commit")
        verify_timer = Timer("verify")
        rail_enforce_timer = Timer("rail_enforce")
        receipt_timer = Timer("receipt")
        render_timer = Timer("render")

        # Check det(gamma) before geometry computation
        from .gr_core_fields import det_sym6
        if np.any(det_sym6(self.fields.gamma_sym6) <= 0):
            return 0.0, "det_gamma_violation", "det(gamma) <= 0"

        # 1. Sense: compute diagnostics
        with sense_timer:
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            max_R = np.max(self.geometry.R) if hasattr(self.geometry, 'R') and self.geometry.R is not None else 0.0
            self.constraints.compute_hamiltonian()
            self.constraint_eval_count_h += 1
            self.constraints.compute_momentum()
            self.constraint_eval_count_m += 1
            self.constraints.compute_residuals()

        eps_H, eps_M = self.constraints.eps_H, self.constraints.eps_M
        from .gr_core_fields import det_sym6, eigenvalues_sym6
        det_gamma = det_sym6(self.fields.gamma_sym6)
        m_det = np.min(det_gamma)
        eigvals = eigenvalues_sym6(self.fields.gamma_sym6)
        lambda_min_early = np.min(eigvals)
        lambda_max_early = np.max(eigvals)
        cond_gamma_early = lambda_max_early / lambda_min_early if lambda_min_early > 0 else np.inf

        # Early rail check with dt shrink loop
        dt = self.scheduler.compute_dt(eps_H, eps_M)  # initial dt
        if dt_max is not None:
            dt = min(dt, dt_max)

        rail_violation = None
        for local_try in range(self.max_attempts_per_step):
            self.attempt_id += 1
            self.tau += 1.0

            eps_H, eps_M, sem_ok, sem_reason = self.sem_safe_compute_residuals()
            rail_violation_early = self.rails.check_gates(eps_H, eps_M, self.geometry, self.fields)

            if rail_violation_early or not sem_ok:
                gate_reason = rail_violation_early if rail_violation_early else sem_reason
                rail_margins = self.rails.compute_margins(eps_H, eps_M, self.geometry, self.fields, self.threads.m_det_min) if rail_violation_early else {"SEM": "hard"}

                # Classify: dt-dependent vs state-dependent
                msg = str(gate_reason)
                kind = "dt" if ("CFL" in msg or "dt" in msg or "timestep" in msg) else "state"

                rec = self.make_attempt_receipt(
                    dt=dt, eps_H=eps_H, eps_M=eps_M,
                    sem_ok=sem_ok, sem_reason=sem_reason,
                    gate_reason=gate_reason,
                    rail_margins=rail_margins,
                    dominant_thread="early_gate" if rail_violation_early else "SEM",
                    accepted=False,
                    action=None,
                    extra={"local_try": local_try, "kind": kind}
                )
                self.M_solve.append(rec)

                if kind == "dt":
                    # dt shrink retry
                    dt_new = dt * self.dt_shrink
                    rec["action"] = {"type": "shrink_dt", "dt_new": float(dt_new)}

                    if dt_new < self.dt_min:
                        rec["action"] = {"type": "abort_dt_floor", "dt_new": float(dt_new)}
                        return dt_new, "early_gate" if rail_violation_early else "SEM", f"{msg}|dt_floor"

                    dt = dt_new
                    continue

                else:
                    # state-dependent: abort immediately
                    rec["action"] = {"type": "abort_state_gate"}
                    return dt, "early_gate" if rail_violation_early else "SEM", msg

            # If pass, proceed to propose
            break
        else:
            return dt, "none", "max_attempts_exceeded", False

        # Compute dt_global from scheduler
        dt_global = self.scheduler.compute_dt(eps_H, eps_M)
        logger.debug("Computed dt_global and regime updates", extra={
            "extra_data": {
                "dt_global": dt_global,
                "loom_active": self.loom_active,
                "dt_loom_prev": self.dt_loom_prev,
                "eps_H": eps_H,
                "eps_M": eps_M,
                "max_R": max_R
            }
        })
        # Update stepper regime hash
        old_stepper_hash = self.clocks.stepper_regime_hash
        self.clocks.update_stepper_regime(eps_H + eps_M, self.eps_H_prev + self.eps_M_prev if self.eps_H_prev is not None else None, self.clocks.rollback_rate)
        if old_stepper_hash != self.clocks.stepper_regime_hash:
            logger.info("Stepper regime shift detected", extra={
                "extra_data": {
                    "old_hash": old_stepper_hash,
                    "new_hash": self.clocks.stepper_regime_hash,
                    "step": self.step,
                    "tau_s": self.clocks.tau_s
                }
            })
            self.aeonic_receipts.emit_event("REGIME_SHIFT", {
                "regime_type": "stepper",
                "hash": self.clocks.stepper_regime_hash,
                "step": self.step,
                "tau_s": self.clocks.tau_s
            })
        # StepperMemory: compute regime hash and suggest dt to adjust dt_global
        regime_hash = self.stepper.memory.compute_regime_hash(eps_H, eps_M, max_R)
        dt_suggested = self.stepper.memory.suggest_dt(regime_hash)
        if self.disable_suggestions_steps > 0:
            dt_suggested = self.stepper.memory.default_dt
        dt_global = min(dt_global, dt_suggested)

        # 2. Propose: each thread proposes dt
        with propose_timer:
            proposals = self.threads.propose_dts(eps_H, eps_M, m_det, self.eps_H_prev, self.eps_M_prev, self.m_det_prev, self.dt_prev, self.geometry, self.gauge, dt_global)

        # Fix dt_thread['PHY_step_observe'] to dt_cfl
        dt_cfl = self.threads.C_CFL * self.fields.dx / 1.0  # c=1
        proposals['PHY_step_observe']['dt'] = dt_cfl

        # Hysteresis for loom activation
        D_max = self.D_max_prev
        if not hasattr(self, 'loom_active'):
            self.loom_active = False
        D_on = 1e-3  # activation threshold
        D_off = 5e-4  # deactivation threshold
        if self.loom_active:
            self.loom_active = (D_max > D_off)
        else:
            self.loom_active = (D_max > D_on)

        # 3. Decide: dt_target = dt_global (capped by dt_max), dt_cap from proposals modified by dt_loom if loom_active, dt_commit = min(dt_target, dt_cap)
        with decide_timer:
            dt_target = dt_global
            if dt_max is not None:
                dt_target = min(dt_target, dt_max)
            dt_cap = self.threads.rho_target * min(p['dt'] for p in proposals.values())
            if self.loom_active and self.dt_loom_prev is not None:
                dt_cap = min(dt_cap, self.dt_loom_prev)
            dt_commit = min(dt_target, dt_cap)
            dt_commit = max(dt_commit, self.dt_min)
            dt = dt_commit
            logger.debug("dt decision made", extra={
                "extra_data": {
                    "dt_target": dt_target,
                    "dt_cap": dt_cap,
                    "dt_commit": dt,
                    "proposals": proposals,
                    "dominant_thread": dominant_thread if 'dominant_thread' in locals() else None
                }
            })
            # Dominant thread is argmin margin among threads with dt
            thread_margins = {k: p['margin'] for k, p in proposals.items() if p['dt'] is not None}
            if self.dt_loom_prev is not None and self.dt_loom_prev > 0:
                loom_margin = 1 - (dt_commit / self.dt_loom_prev) if self.dt_loom_prev > 0 else 1.0
                thread_margins['loom'] = loom_margin
            dominant_thread = min(thread_margins, key=thread_margins.get) if thread_margins else 'none'

        # 4. Predict: cheap trial step (placeholder, skip for now)

        # 5. Commit: real evolution step
        with commit_timer:
            # Pre-commit audit for evolve op (requires_audit_before_commit=True)
            eps_H_pre_commit, eps_M_pre_commit, sem_ok_pre, sem_reason_pre = self.sem_safe_compute_residuals()
            if not sem_ok_pre:
                # Audit failed, reject commit and rollback
                self.rollback_count += 1
                self.rollback_reason = f"Pre-commit audit failed: {sem_reason_pre}"
                logger.warning("Pre-commit audit failed for evolve op, rolling back", extra={
                    "extra_data": {
                        "step": self.step,
                        "sem_reason": sem_reason_pre,
                        "eps_H": eps_H_pre_commit,
                        "eps_M": eps_M_pre_commit
                    }
                })
                # No state to restore yet, since we haven't modified
                return dt, "SEM", sem_reason_pre

            # Save state for potential rollback
            state_backup = {
                'gamma': self.fields.gamma_sym6.copy(),
                'K': self.fields.K_sym6.copy(),
                'alpha': self.fields.alpha.copy(),
                'beta': self.fields.beta.copy(),
                'phi': self.fields.phi.copy()
            }

            t_prev = self.t
            substeps = 0
            dt_applied = dt
            if dt_commit < dt_target:
                N = int(np.ceil(dt_target / dt_commit))
                dt_sub = dt_target / N
                substeps = N
                dt_applied = dt_sub
                logger.debug("Substepping activated", extra={
                    "extra_data": {
                        "N_substeps": N,
                        "dt_sub": dt_sub,
                        "dt_target": dt_target,
                        "dt_commit": dt_commit
                    }
                })
                for i_sub in range(N):
                    self.stepper.step_ufe(dt_sub)
                    self.gauge.evolve_lapse(dt_sub)
                    self.gauge.evolve_shift(dt_sub)
                dt = dt_target  # advance by dt_target
            else:
                self.stepper.step_ufe(dt, self.t)
                self.gauge.evolve_lapse(dt)
                self.gauge.evolve_shift(dt)
                # dt is dt_commit = dt_target

            # PhaseLoom scheduling
            K_current = self.fields.K_sym6[..., 0]
            gamma_current = self.fields.gamma_sym6[..., 0]
            residual_slope = eps_H  # Using current eps_H as proxy for residual slope
            rollback_occurred = (self.rollback_count > rollback_count_prev)
            should_compute = self.loom_memory.should_compute_loom(self.step, K_current, gamma_current, residual_slope, rollback_occurred)

            if should_compute:
                # Aeonic PhaseLoom: extract signals, compute coherence
                a_vals, b_vals = self.adapter.extract_thread_signals(self.fields, self.constraints, self.geometry)
                theta, rho, omega_phase = self.adapter.compute_theta_rho_omega(a_vals, b_vals, self.t + dt, dt)
                # Compute spectral omega for Loom activation
                spectral_omega = compute_omega_current(self.fields, self.prev_K, self.prev_gamma, self.spectral_cache)
                loom_data = self.octaves.process_sample(spectral_omega)
                logger.debug("Loom computation details", extra={
                    "extra_data": {
                        "step": self.step,
                        "loom_active": self.loom_active,
                        "D_max": loom_data.get('D_max', 0.0),
                        "delta_K_inf": float(np.max(np.abs(self.fields.K_sym6[...,0] - self.prev_K))) if self.prev_K is not None else None,
                        "K_inf": float(np.max(np.abs(self.fields.K_sym6[...,0])))
                    }
                }) if self.prev_K is not None else None
                dt_loom, mu_scale = self.controller.get_controls(loom_data)
                self.dt_loom_prev = dt_loom
                self.D_max_prev = loom_data.get('D_max', 0.0)
                # Tick loom clock and update regime
                self.clocks.tick_loom_update()
                dominant_band = np.argmax(loom_data['D_band'])
                centroid = loom_data.get('centroid', 0.0)  # Assume centroid is computed in loom_data
                transfer_pattern = 0  # Placeholder
                old_loom_hash = self.clocks.loom_regime_hash
                self.clocks.update_loom_regime(self.D_max_prev, dominant_band, centroid, transfer_pattern)
                if old_loom_hash != self.clocks.loom_regime_hash:
                    self.aeonic_receipts.emit_event("REGIME_SHIFT", {
                        "regime_type": "loom",
                        "hash": self.clocks.loom_regime_hash,
                        "step": self.step,
                        "tau_l": self.clocks.tau_l
                    })

                # Apply Aeonic mu updates
                self.threads.mu_H *= mu_scale
                self.threads.mu_M *= mu_scale

                self.loom_memory.post_loom_update(loom_data)
                loom_computed = True
            else:
                loom_data = self.loom_memory.summary
                dt_loom = self.dt_loom_prev if self.dt_loom_prev is not None else dt
                mu_scale = 1.0  # No update if not computed
                logger.debug("Loom computation skipped", extra={
                    "extra_data": {
                        "step": self.step
                    }
                })
                loom_computed = False

            # Update loom memory prev for next check
            self.loom_memory.prev_K = K_current
            self.loom_memory.prev_gamma = gamma_current
            self.loom_memory.prev_residual_slope = residual_slope

        # 6. Verify: recompute constraints + residuals
        with verify_timer:
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.constraints.compute_hamiltonian()
            self.constraint_eval_count_h += 1
            self.constraints.compute_momentum()
            self.constraint_eval_count_m += 1
            self.constraints.compute_residuals()
            eps_H_post, eps_M_post = self.constraints.eps_H, self.constraints.eps_M
            logger.debug("Residuals computed", extra={
                "extra_data": {
                    "eps_H_pre": eps_H,
                    "eps_M_pre": eps_M,
                    "eps_H_post": eps_H_post,
                    "eps_M_post": eps_M_post
                }
            })
            eps_UFE = 0.0  # Placeholder
            consistency_ok = bool(eps_H_post < self.threads.eps_H_target and eps_M_post < self.threads.eps_M_target)

        # Compute gamma diagnostics
        from .gr_core_fields import det_sym6, eigenvalues_sym6
        det_gamma = det_sym6(self.fields.gamma_sym6)
        eigvals = eigenvalues_sym6(self.fields.gamma_sym6)
        lambda_min = np.min(eigvals)
        lambda_max = np.max(eigvals)
        cond_gamma = lambda_max / lambda_min if lambda_min > 0 else np.inf

        # 7. Rail-enforce: check gates
        with rail_enforce_timer:
            rail_violation = self.rails.check_gates(eps_H_post, eps_M_post, self.geometry, self.fields)
            rail_margins = self.rails.compute_margins(eps_H_post, eps_M_post, self.geometry, self.fields, self.threads.m_det_min)
            repair_applied = False
            repair_type = None
            lambda_min_pre = None
            lambda_min_post = None
            if rail_violation and "lambda_min" in rail_violation:
                # Attempt SPD repair
                from .gr_core_fields import repair_spd_eigen_clamp
                repaired_gamma, lambda_min_pre_val, lambda_min_post_val = repair_spd_eigen_clamp(self.fields.gamma_sym6, self.rails.lambda_floor)
                lambda_min_pre = lambda_min_pre_val
                lambda_min_post = lambda_min_post_val
                # Apply repair
                self.fields.gamma_sym6 = repaired_gamma
                repair_applied = True
                repair_type = "spd_eig_clamp"
                # Recompute geometry and constraints after repair
                self.geometry.compute_christoffels()
                self.geometry.compute_ricci()
                self.geometry.compute_scalar_curvature()
                self.constraints.compute_hamiltonian()
                self.constraint_eval_count_h += 1
                self.constraints.compute_momentum()
                self.constraint_eval_count_m += 1
                self.constraints.compute_residuals()
                eps_H_repaired = self.constraints.eps_H
                eps_M_repaired = self.constraints.eps_M
                # Check if admissibility restored and residuals not exploded
                rail_violation_repaired = self.rails.check_gates(eps_H_repaired, eps_M_repaired, self.geometry, self.fields)
                residual_exploded = (eps_H_repaired > 10 * eps_H_post) or (eps_M_repaired > 10 * eps_M_post)
                if rail_violation_repaired is None and not residual_exploded:
                    # Accept repair
                    eps_H_post = eps_H_repaired
                    eps_M_post = eps_M_repaired
                    rail_violation = None  # Clear violation
                    logger.info("SPD repair applied successfully", extra={
                        "extra_data": {
                            "step": self.step,
                            "repair_type": repair_type
                        }
                    })
                else:
                    # Repair failed, rollback
                    self.rollback_count += 1
                    self.rollback_reason = f"SPD repair failed: {rail_violation_repaired or 'residual exploded'}"
                    logger.warning("SPD repair failed, rolling back", extra={
                        "extra_data": {
                            "step": self.step,
                            "rollback_reason": self.rollback_reason,
                            "repair_type": repair_type
                        }
                    })
                    
                    # Restore state
                    self.fields.gamma_sym6[:] = state_backup['gamma']
                    self.fields.K_sym6[:] = state_backup['K']
                    self.fields.alpha[:] = state_backup['alpha']
                    self.fields.beta[:] = state_backup['beta']
                    self.fields.phi[:] = state_backup['phi']
                    
                    # Signal violation to caller (who may break or handle it)
                    # Note: In a full implementation, we would retry here with dt*0.5
            elif rail_violation:
                # Other violations: rollback as before
                self.rollback_count += 1
                self.rollback_reason = rail_violation
                logger.warning("Rail violation detected", extra={
                    "extra_data": {
                        "step": self.step,
                        "rail_violation": rail_violation,
                        "rollback_count": self.rollback_count
                    }
                })
                
                # Restore state
                self.fields.gamma_sym6[:] = state_backup['gamma']
                self.fields.K_sym6[:] = state_backup['K']
                self.fields.alpha[:] = state_backup['alpha']
                self.fields.beta[:] = state_backup['beta']
                self.fields.phi[:] = state_backup['phi']
            else:
                # Emit warnings if margins exceed threshold
                for rail, margin in rail_margins.items():
                    if margin > self.rails.warning_threshold:
                        self.receipts.emit_rail_warning(rail, margin)

        # StepperMemory: post_step_update
        success = (self.rollback_count == rollback_count_prev)
        self.stepper.memory.post_step_update(dt, success, eps_H + eps_M, eps_H_post + eps_M_post, regime_hash)
        # Honesty check and mark tainted if violation
        honesty_ok = self.stepper.memory.honesty_ok(eps_H + eps_M, eps_H_post + eps_M_post, self.rails, eps_H_post, eps_M_post, self.geometry, self.fields)
        if not honesty_ok:
            self.tainted = True
            self.rollback_count += 1
            self.rollback_reason = "Tainted by honesty check"
            logger.error("Tainted by honesty check", extra={
                "extra_data": {
                    "step": self.step,
                    "violation": not honesty_ok,
                    "regime_hash": regime_hash
                }
            })
            self.memory.mark_tainted(regime_hash)
            self.disable_suggestions_steps = 5
        else:
            self.tainted = False

        # Loom honesty check
        loom_violation = not loom_computed and self.loom_memory.honesty_check(True)
        if loom_violation:
            self.tainted = True
            self.rollback_count += 1
            self.rollback_reason = "Loom skipped but D_max high"
            logger.error("Tainted by loom honesty check", extra={
                "extra_data": {
                    "step": self.step,
                    "loom_violation": loom_violation,
                    "D_max_prev": self.loom_memory.D_max_prev
                }
            })
            self.memory.mark_tainted(self.clocks.loom_regime_hash)
            self.disable_suggestions_steps = 5

        final_success = (self.rollback_count == rollback_count_prev)
        if not final_success:
            logger.warning("Rollback occurred", extra={
                "extra_data": {
                    "step": self.step,
                    "rollback_reason": self.rollback_reason,
                    "rollback_count": self.rollback_count
                }
            })
        if final_success:
            self.accepted_step_count += 1
            if self.accepted_step_count % 8 == 0:
                self.memory.maintenance_tick()

        # 8. Receipt: emit Ω-ledger + PhaseLoom event
        with receipt_timer:
            alpha_min, alpha_max = np.min(self.fields.alpha), np.max(self.fields.alpha)
            max_R = np.max(self.geometry.R) if hasattr(self.geometry, 'R') and self.geometry.R is not None else 0.0
            t_expected = self.t_expected
            t_err = self.t - t_expected
            risk_gauge = self.scheduler.compute_risk_gauge(proposals, dt, self.dt_loom_prev)
            loom_margin = 1 - (dt / self.dt_loom_prev) if self.dt_loom_prev is not None and self.dt_loom_prev > 0 else 1.0
            loom_ratio = dt / self.dt_loom_prev if self.dt_loom_prev is not None and self.dt_loom_prev > 0 else 0.0
            # Tight threads: top 3 active by smallest margin
            thread_list = [(k, p['ratio'], p['margin']) for k, p in proposals.items() if p['active']]
            loom_active = self.dt_loom_prev is not None and self.dt_loom_prev > 0
            if loom_active:
                thread_list.append(('loom', loom_ratio, loom_margin))
            tight_threads = sorted(thread_list, key=lambda x: x[2])[:3]
            # Compute rate of change for constraints
            d_eps_H = (eps_H_post - self.eps_H_prev) / dt if self.eps_H_prev is not None else 0.0
            d_eps_M = (eps_M_post - self.eps_M_prev) / dt if self.eps_M_prev is not None else 0.0
            self.receipts.emit_m_solve(self.step, self.t, dt, dominant_thread, proposals, eps_H, eps_M, eps_H_post, eps_M_post, d_eps_H, d_eps_M, max_R, m_det, self.threads.mu_H, self.threads.mu_M, self.rollback_count, self.rollback_reason, loom_data, t_expected, t_err, self.dt_loom_prev, risk_gauge, tight_threads, consistency_ok, rail_margins, lambda_min, lambda_max, cond_gamma, repair_applied, repair_type, lambda_min_pre, lambda_min_post, t_prev, self.t, dt, dt_applied, substeps, True, self.policy_hash)

        # 9. Render: push to visualization buffers
        with render_timer:
            self.render.update_channels()

        self.t += dt
        self.t_expected += dt

        # Time commit rail
        time_tol = 1e-12
        if abs((self.t - t_prev) - dt) > time_tol:
            raise AssertionError(f"Time commit failed: t_prev={t_prev}, t={self.t}, dt={dt}")

        self.step += 1

        # Emit M_step receipt
        if self.receipts.config.enable_M_step:
            self.receipts.emit_m_step(self.step, self.t, dt, dominant_thread, proposals, eps_H, eps_M, eps_H_post, eps_M_post, d_eps_H, d_eps_M, max_R, m_det, self.threads.mu_H, self.threads.mu_M, self.rollback_count, self.rollback_reason, None, t_expected, t_err, self.dt_loom_prev, None, None, consistency_ok, None, lambda_min, lambda_max, cond_gamma, repair_applied, repair_type, None, None, t_prev, self.t, dt, dt_applied, substeps, True, self.policy_hash)

        # Emit macro receipt every K steps
        if self.step % self.receipts.config.K == 0:
            self.receipts.emit_macro(self.step)

        # Update previous for next step
        self.eps_H_prev = eps_H_post
        self.eps_M_prev = eps_M_post
        self.m_det_prev = m_det
        self.dt_prev = dt
        self.prev_K = self.fields.K_sym6[..., 0].copy()
        self.prev_gamma = self.fields.gamma_sym6[..., 0].copy()

        if self.disable_suggestions_steps > 0:
            self.disable_suggestions_steps -= 1

        # Log detailed timing breakdown
        logger.info("Step timing breakdown", extra={
            "extra_data": {
                "sense_ms": sense_timer.elapsed_ms(),
                "propose_ms": propose_timer.elapsed_ms(),
                "decide_ms": decide_timer.elapsed_ms(),
                "commit_ms": commit_timer.elapsed_ms(),
                "verify_ms": verify_timer.elapsed_ms(),
                "rail_enforce_ms": rail_enforce_timer.elapsed_ms(),
                "receipt_ms": receipt_timer.elapsed_ms(),
                "render_ms": render_timer.elapsed_ms(),
                "total_ms": sense_timer.elapsed_ms() + propose_timer.elapsed_ms() + decide_timer.elapsed_ms() + commit_timer.elapsed_ms() + verify_timer.elapsed_ms() + rail_enforce_timer.elapsed_ms() + receipt_timer.elapsed_ms() + render_timer.elapsed_ms()
            }
        })

        return dt, dominant_thread, rail_violation