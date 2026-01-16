# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\Psi": "UFE_state"
}

import numpy as np
import logging
from .logging_config import setup_logging, Timer, array_stats
from .gr_core_fields import GRCoreFields, SYM6_IDX
from .gr_geometry import GRGeometry
from .gr_constraints import GRConstraints
from .gr_gauge import GRGauge
from .gr_scheduler import GRScheduler
from .gr_stepper import GRStepper
from .gr_ledger import GRLedger
from .phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_memory_contract import AeonicMemoryContract
from aeonic_clocks import AeonicClockPack
from aeonic_receipts import AeonicReceipts
from phaseloom_27 import PhaseLoom27

logger = logging.getLogger('gr_solver.solver')

class GRSolver:
    def __init__(self, Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0, c=1.0, Lambda=0.0, log_level=logging.INFO, log_file=None):
        # Setup structured logging
        setup_logging(level=log_level, log_file=log_file)

        logger.info("Initializing GR Solver", extra={
            "extra_data": {
                "grid_size": [Nx, Ny, Nz],
                "grid_spacing": [dx, dy, dz],
                "c": c,
                "Lambda": Lambda
            }
        })

        self.fields = GRCoreFields(Nx, Ny, Nz, dx, dy, dz)
        self.geometry = GRGeometry(self.fields)
        self.constraints = GRConstraints(self.fields, self.geometry)
        self.gauge = GRGauge(self.fields, self.geometry)
        self.scheduler = GRScheduler(self.fields, c, Lambda)

        # Aeonic Memory and PhaseLoom
        self.clocks = AeonicClockPack()
        self.aeonic_receipts = AeonicReceipts()
        self.memory_bank = AeonicMemoryBank(self.clocks, self.aeonic_receipts)
        self.memory_contract = AeonicMemoryContract(self.memory_bank, self.aeonic_receipts)
        self.phaseloom = PhaseLoom27()

        self.stepper = GRStepper(self.fields, self.geometry, self.constraints, self.gauge, self.memory_contract, self.phaseloom)
        self.ledger = GRLedger()
        self.orchestrator = GRPhaseLoomOrchestrator(self.fields, self.geometry, self.constraints, self.gauge, self.stepper, self.ledger, self.memory_contract, self.phaseloom)
        self.t = 0.0
        self.step = 0

        logger.info("GR Solver initialized successfully")

    def init_minkowski(self):
        self.fields.init_minkowski()
        # Inject high-k ripple into gamma_xx to excite upper spectral bands and trigger Loom activity
        epsilon = 1e-6
        k = 10.0  # high wavenumber
        x = np.arange(self.fields.Nx) * self.fields.dx - (self.fields.Nx * self.fields.dx) / 2
        y = np.arange(self.fields.Ny) * self.fields.dy - (self.fields.Ny * self.fields.dy) / 2
        z = np.arange(self.fields.Nz) * self.fields.dz - (self.fields.Nz * self.fields.dz) / 2
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.fields.gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
        # Add separate random perturbation to gamma_sym6 to make initial eps_H and eps_M nonzero
        eps_metric = 1e-6
        self.fields.gamma_sym6 += eps_metric * np.random.randn(*self.fields.gamma_sym6.shape)
        # Compute initial geometry and constraints
        self.geometry.compute_christoffels()
        self.geometry.compute_ricci()
        self.geometry.compute_scalar_curvature()
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()
        self.constraints.compute_residuals()

    def run(self, T_max, dt_max=None):
        """Main PhaseLoom evolution loop."""
        logger.info("Starting GR solver evolution", extra={
            "extra_data": {
                "T_max": T_max,
                "dt_max": dt_max,
                "initial_t": self.t,
                "initial_step": self.step
            }
        })

        total_timer = Timer("total_evolution")
        with total_timer:
            while self.orchestrator.t < T_max:
                step_timer = Timer("step")
                with step_timer:
                    dt, dominant_thread, rail_violation = self.orchestrator.run_step(dt_max)

                self.t = self.orchestrator.t
                self.step = self.orchestrator.step

                # Log step metrics
                logger.info("Evolution step completed", extra={
                    "extra_data": {
                        "step": self.step,
                        "t": self.t,
                        "dt": dt,
                        "dominant_thread": dominant_thread,
                        "step_execution_time_ms": step_timer.elapsed_ms(),
                        "rail_violation": rail_violation,
                        "constraint_residuals": {
                            "eps_H": float(self.constraints.eps_H) if hasattr(self.constraints, 'eps_H') else None,
                            "eps_M": float(self.constraints.eps_M) if hasattr(self.constraints, 'eps_M') else None
                        },
                        "dominant_clock": getattr(self.clocks, 'dominant_clock', None),
                        "field_stats": {
                            "alpha": array_stats(self.fields.alpha, "alpha"),
                            "K_trace": array_stats(trace_sym6(self.fields.K_sym6, inv_sym6(self.fields.gamma_sym6)), "K_trace")
                        }
                    }
                })

                # Optional: break on violation or max steps
                if rail_violation:
                    logger.warning("Stopping due to rail violation", extra={
                        "extra_data": {
                            "rail_violation": rail_violation,
                            "step": self.step,
                            "t": self.t
                        }
                    })
                    break

        logger.info("Evolution completed", extra={
            "extra_data": {
                "total_steps": self.step,
                "final_t": self.t,
                "total_execution_time_ms": total_timer.elapsed_ms(),
                "average_step_time_ms": total_timer.elapsed_ms() / max(1, self.step)
            }
        })