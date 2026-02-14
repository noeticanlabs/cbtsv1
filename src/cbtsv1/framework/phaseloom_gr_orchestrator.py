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
import json
from src.aeonic.aeonic_clocks import AeonicClockPack
from src.aeonic.aeonic_memory_bank import AeonicMemoryBank
from src.core.aeonic_receipts import AeonicReceipts
from src.core.aeonic_memory_contract import AeonicMemoryContract
from src.core.logging_config import Timer, array_stats
from .phaseloom_threads_gr import GRPhaseLoomThreads, compute_omega_current
from src.core.phases import (
    PhaseState, SensePhase, ProposePhase, DecidePhase, PredictPhase,
    CommitPhase, VerifyPhase, RailEnforcePhase, ReceiptPhase, RenderPhase
)
from .phaseloom_rails_gr import GRPhaseLoomRails
from .phaseloom_receipts_gr import GRPhaseLoomReceipts
from .phaseloom_render_gr import GRPhaseLoomRender
from .phaseloom_gr_adapter import GRPhaseLoomAdapter
from .phaseloom_octaves import PhaseLoomOctaves
from .phaseloom_gr_controller import GRPhaseLoomController
from src.core.gr_scheduler import GRScheduler
from src.core.gr_sem import SEMDomain
from .phaseloom_memory import PhaseLoomMemory
from src.core.gr_ttl_calculator import TTLCalculator
from src.receipts.receipt_schemas import Kappa
from src.receipts.orchestrator_contract_memory import OrchestratorContractWithMemory
from src.nllc.vm import VM
from src.nllc.nir import Module, Function, BasicBlock, ConstInst, BinOpInst, CallInst, BrInst, RetInst, Value, Type, Trace, Span

logger = logging.getLogger('gr_solver.orchestrator')

def load_module_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    functions = []
    for f_data in data['functions']:
        params = []
        for p in f_data['params']:
            params.append(Value(p['name'], Type()))
        return_ty = Type()
        blocks = []
        for b_data in f_data['blocks']:
            instructions = []
            for i_data in b_data['instructions']:
                trace = Trace(i_data['trace']['file'], Span(i_data['trace']['span']['start'], i_data['trace']['span']['end']), i_data['trace']['ast_path'])
                if 'value' in i_data and 'op' not in i_data and 'func' not in i_data and 'result' in i_data and i_data['result'] is not None:
                    result = Value(i_data['result']['name'], Type())
                    inst = ConstInst(trace, result, i_data['value'])
                elif 'op' in i_data and 'left' in i_data:
                    result = Value(i_data['result']['name'], Type())
                    left = Value(i_data['left']['name'], Type())
                    right = Value(i_data['right']['name'], Type())
                    inst = BinOpInst(trace, result, left, i_data['op'], right)
                elif 'func' in i_data:
                    if 'result' in i_data and i_data['result']:
                        result = Value(i_data['result']['name'], Type())
                    else:
                        result = None
                    args = [Value(a['name'], Type()) for a in i_data['args']]
                    inst = CallInst(trace, result, i_data['func'], args)
                elif 'cond' in i_data:
                    cond = Value(i_data['cond']['name'], Type()) if i_data.get('cond') else None
                    inst = BrInst(trace, cond, i_data['true_block'], i_data.get('false_block'))
                elif 'value' in i_data and i_data.get('value') is None:
                    value = None
                    if 'value' in i_data and i_data['value']:
                        value = Value(i_data['value']['name'], Type())
                    inst = RetInst(trace, value)
                elif 'array' in i_data:
                    result = Value(i_data['result']['name'], Type())
                    array = Value(i_data['array']['name'], Type())
                    index = Value(i_data['index']['name'], Type())
                    from src.nllc.nir import GetElementInst
                    inst = GetElementInst(trace, result, array, index)
                else:
                    inst = None  # skip unknown
                if inst:
                    instructions.append(inst)
            blocks.append(BasicBlock(b_data['name'], instructions))
        functions.append(Function(f_data['name'], params, return_ty, blocks))
    return Module(functions)


class GRHostAPI:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def gr_snapshot(self):
        state = {
            'gamma': self.orchestrator.fields.gamma_sym6.copy(),
            'K': self.orchestrator.fields.K_sym6.copy(),
            'alpha': self.orchestrator.fields.alpha.copy(),
            'beta': self.orchestrator.fields.beta.copy(),
            'phi': self.orchestrator.fields.phi.copy()
        }
        return state

    def gr_step(self, dt, t):
        accepted, _, _, reason = self.orchestrator.stepper.step_ufe(dt, t)
        return accepted, reason

    def gr_apply_gauge(self, dt):
        self.orchestrator.gauge.evolve_lapse(dt)
        self.orchestrator.gauge.evolve_shift(dt)

    def gr_compute_constraints(self):
        self.orchestrator.constraints.compute_hamiltonian()
        self.orchestrator.constraint_eval_count_h += 1
        self.orchestrator.constraints.compute_momentum()
        self.orchestrator.constraint_eval_count_m += 1
        self.orchestrator.constraints.compute_residuals()
        return {'eps_H': float(self.orchestrator.constraints.eps_H), 'eps_M': float(self.orchestrator.constraints.eps_M)}

    def gr_accept_step(self):
        self.orchestrator.aeonic_receipts.emit_event('STEP_ACCEPTED', {'step': self.orchestrator.step, 't': self.orchestrator.t})

    def gr_restore(self, snapshot):
        self.orchestrator.fields.gamma_sym6[:] = snapshot['gamma']
        self.orchestrator.fields.K_sym6[:] = snapshot['K']
        self.orchestrator.fields.alpha[:] = snapshot['alpha']
        self.orchestrator.fields.beta[:] = snapshot['beta']
        self.orchestrator.fields.phi[:] = snapshot['phi']

    def gr_reject_step(self):
        self.orchestrator.rollback_count += 1
        self.orchestrator.rollback_reason = 'VM rejected'
        self.orchestrator.aeonic_receipts.emit_event('STEP_REJECTED', {'step': self.orchestrator.step, 't': self.orchestrator.t, 'reason': 'VM constraint check failed'})

    def print(self, *args):
        print(*args)

class GRPhaseLoomOrchestrator:
    def __init__(self, fields, geometry, constraints, gauge, stepper, ledger, memory_contract=None, phaseloom=None, eps_H_target=1e-8, eps_M_target=1e-8, m_det_min=0.2, aeonic_mode=True):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.stepper = stepper
        self.ledger = ledger
        self.aeonic_mode = aeonic_mode

        self.threads = GRPhaseLoomThreads(fields, eps_H_target, eps_M_target, m_det_min)
        self.eps_H_target = eps_H_target
        self.eps_M_target = eps_M_target
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
            
            # Create TTL calculator with simulation parameters (can be updated later)
            self.ttl_calculator = None  # Set via set_simulation_params()
            
            self.memory_contract = AeonicMemoryContract(
                self.memory_bank, 
                self.aeonic_receipts,
                ttl_calculator=self.ttl_calculator
            )
        else:
            self.memory_contract = memory_contract
            self.memory_bank = self.memory_contract.memory_bank  # for backward compatibility
            self.clocks = self.memory_bank.clock
            self.aeonic_receipts = self.memory_bank.receipts
            self.ttl_calculator = getattr(self.memory_contract, 'ttl_calculator', None)
        self.memory = self.memory_bank  # alias for backward compatibility
        self.orchestrator_memory = OrchestratorContractWithMemory(self.memory_contract)
        
        # Initialize clock system first, then pass to PhaseLoomMemory
        self.clock_system = AeonicClockPack()
        self.loom_memory = PhaseLoomMemory(self.fields, clock_system=self.clock_system, base_dt=0.001)

        # SEM Domain
        self.sem_domain = SEMDomain()

        # Spectral Cache
        from src.spectral.cache import SpectralCache
        self.spectral_cache = SpectralCache(self.fields.Nx, self.fields.Ny, self.fields.Nz, self.fields.dx, self.fields.dy, self.fields.dz)
        # Cache in Tier3
        bytes_est = (self.spectral_cache.kx.nbytes + self.spectral_cache.ky.nbytes + self.spectral_cache.kz.nbytes +
                     self.spectral_cache.k2.nbytes + self.spectral_cache.dealias_mask.nbytes +
                     self.spectral_cache.kx_bin.nbytes + self.spectral_cache.ky_bin.nbytes + self.spectral_cache.kz_bin.nbytes)
        self.memory.put("spectral_cache", 3, self.spectral_cache, bytes_est, ttl_l=1000000, ttl_s=1000000, recompute_cost_est=1000.0, risk_score=0.0, tainted=False, regime_hashes=[])

        # Load NLLC NIR module and instantiate VM
        try:
            self.nir_module = load_module_from_json('compiled_nir.json')
            self.gr_host_api = GRHostAPI(self)
            self.vm = VM(self.nir_module, module_id='gr_solver_nllc', dep_closure_hash='dep_hash', gr_host_api=self.gr_host_api)
        except (FileNotFoundError, json.JSONDecodeError, ImportError, Exception) as e:
            logger.warning(f"Could not load compiled_nir.json or VM dependencies, VM disabled: {e}")
            self.nir_module = None
            self.vm = None

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

        # Simulation parameters for adaptive TTL
        self.t_end = None  # Final simulation time
        self.dt_avg = None  # Average timestep
        self.N = None  # Grid size

        # Aeonic counters
        self.attempt_id = 0
        self.step_id = 0
        self.tau = 0.0

        # dt policy
        self.dt_min = 1e-8
        self.dt_max = None
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
        self.prev_omega_current = None
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

    def set_simulation_params(self, t_end: float, dt_avg: float, problem_type: str = 'standard'):
        """
        Set simulation parameters for adaptive TTL calculation.
        
        Args:
            t_end: Final simulation time
            dt_avg: Average timestep
            problem_type: Type of simulation ('standard', 'long_run', 'high_frequency', 'critical', 'transient')
        """
        self.t_end = t_end
        self.dt_avg = dt_avg
        self.problem_type = problem_type
        
        # Compute effective grid size (Nx * Ny * Nz or max dimension)
        if hasattr(self.fields, 'Nx') and hasattr(self.fields, 'Ny') and hasattr(self.fields, 'Nz'):
            self.N = max(self.fields.Nx, self.fields.Ny, self.fields.Nz)
        else:
            self.N = 64  # Default fallback
        
        # Create and set the TTL calculator
        self.ttl_calculator = TTLCalculator(
            t_end=t_end,
            dt_avg=dt_avg,
            N=self.N,
            problem_type=problem_type
        )
        
        # Update the memory contract's TTL calculator
        if self.memory_contract is not None:
            self.memory_contract.set_ttl_calculator(self.ttl_calculator)
        
        logger.info(f"Simulation params set: t_end={t_end}, dt_avg={dt_avg}, N={self.N}, problem_type={problem_type}")

    def run_step(self, dt_max=None):
        """Execute one PhaseLoom step: Sense, Propose, Decide, Predict, Commit, Verify, Rail-enforce, Receipt, Render"""

        logger.debug("Starting modular orchestrator run_step", extra={
            "extra_data": {
                "step": self.step,
                "t": self.t,
                "rollback_count": self.rollback_count
            }
        })
        rollback_count_prev = self.rollback_count

        original_dt_max = self.dt_max
        try:
            # Set per-step dt_max override if provided
            if dt_max is not None:
                self.dt_max = dt_max

            # Initialize phase state
            state = PhaseState(self)
            state.rollback_count_prev = rollback_count_prev

            # Define phases
            phases = [
                SensePhase(),
                ProposePhase(),
                DecidePhase(),
                PredictPhase(),
                CommitPhase(),
                VerifyPhase(),
                RailEnforcePhase(),
                ReceiptPhase(),
                RenderPhase()
            ]

            # Execute phases sequentially
            for phase in phases:
                print(f"Starting {phase.__class__.__name__}")
                state = phase.execute(state)
                print(f"Finished {phase.__class__.__name__}")
                # Early exit on violation
                if state.rail_violation and state.rail_violation in ["det_gamma_violation", "SEM"]:
                    return state.dt or 0.0, state.dominant_thread or "none", state.rail_violation

            # Final updates from state
            final_success = state.final_success if state.final_success is not None else (state.rollback_count == state.rollback_count_prev)
            if final_success:
                state.orchestrator.t += state.dt
                state.orchestrator.t_expected += state.dt
                state.orchestrator.step += 1
                state.orchestrator.accepted_step_count += 1
                if state.orchestrator.accepted_step_count % 8 == 0:
                    state.orchestrator.memory.maintenance_tick()
        finally:
            # Restore original dt_max
            self.dt_max = original_dt_max

        # Update previous values
        if state.eps_H is not None:
            state.orchestrator.eps_H_prev = state.eps_H
        if state.eps_M is not None:
            state.orchestrator.eps_M_prev = state.eps_M
        if state.m_det is not None:
            state.orchestrator.m_det_prev = state.m_det
        state.orchestrator.dt_prev = state.dt
        if hasattr(state.orchestrator.fields, 'K_sym6') and state.orchestrator.fields.K_sym6.shape[-1] > 0:
            state.orchestrator.prev_K = state.orchestrator.fields.K_sym6[..., 0].copy()
        if hasattr(state.orchestrator.fields, 'gamma_sym6') and state.orchestrator.fields.gamma_sym6.shape[-1] > 0:
            state.orchestrator.prev_gamma = state.orchestrator.fields.gamma_sym6[..., 0].copy()

        if state.orchestrator.disable_suggestions_steps > 0:
            state.orchestrator.disable_suggestions_steps -= 1

        # Store last state for auditing
        self.last_state = state

        # Log timing
        total_ms = sum(timer.elapsed_ms() for timer in [state.sense_timer, state.propose_timer, state.decide_timer, state.commit_timer, state.verify_timer, state.rail_enforce_timer, state.receipt_timer, state.render_timer])
        logger.info("Modular step timing breakdown", extra={
            "extra_data": {
                "sense_ms": state.sense_timer.elapsed_ms(),
                "propose_ms": state.propose_timer.elapsed_ms(),
                "decide_ms": state.decide_timer.elapsed_ms(),
                "commit_ms": state.commit_timer.elapsed_ms(),
                "verify_ms": state.verify_timer.elapsed_ms(),
                "rail_enforce_ms": state.rail_enforce_timer.elapsed_ms(),
                "receipt_ms": state.receipt_timer.elapsed_ms(),
                "render_ms": state.render_timer.elapsed_ms(),
                "total_ms": total_ms
            }
        })

        return state.dt or 0.0, state.dominant_thread or "none", state.rail_violation
