import numpy as np
import logging
from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gcat1c_intentional_failure():
    """GCAT-1C: Intentional failure and rollback for det(gamma) <= 0 violation."""

    # Setup solver with small grid for speed
    solver = GRSolver(Nx=8, Ny=8, Nz=8, dx=1.0, dy=1.0, dz=1.0)
    solver.init_minkowski()

    # Run a few steps to stabilize
    num_initial_steps = 3
    for step in range(num_initial_steps):
        dt, dominant_thread, rail_violation = solver.orchestrator.run_step()
        logger.info(f"Initial step {step}: dt={dt:.6f}, dominant={dominant_thread}, rollback_count={solver.orchestrator.rollback_count}")

    # Ensure no rollbacks so far
    assert solver.orchestrator.rollback_count == 0, f"Unexpected rollbacks before injection: {solver.orchestrator.rollback_count}"

    # Inject intentional violation: set det(gamma) <= 0 by making gamma_xx negative
    solver.fields.gamma_sym6[0, 0, 0, SYM6_IDX["xx"]] = -0.1
    logger.info("Injected violation: gamma_sym6[0,0,0,xx] = -0.1")

    # Run one more step to trigger rollback
    dt, dominant_thread, rail_violation = solver.orchestrator.run_step()
    logger.info(f"Violation step: dt={dt:.6f}, dominant={dominant_thread}, rail_violation={rail_violation}, rollback_count={solver.orchestrator.rollback_count}")

    # Get the receipt for the violating step
    receipts = solver.orchestrator.receipts.receipts
    violation_receipt = receipts[-1]  # Last receipt

    # Assertions
    assert solver.orchestrator.rollback_count > 0, "Rollback did not occur"
    assert violation_receipt["rails"]["reason"] == "det(gamma) <= 0", f"Violation reason mismatch: {violation_receipt['rails']['reason']}"
    assert "det" in violation_receipt["rails"]["margins"], "Triggering rail 'det' not in margins"
    assert "det_gamma_min" in violation_receipt["geometry"], "Receipt missing det_gamma_min"
    assert violation_receipt["geometry"]["det_gamma_min"] <= 0, f"det_gamma_min not negative: {violation_receipt['geometry']['det_gamma_min']}"
    # Rollback within 1 step: since injected just before the step, it triggers in the same step

    logger.info("GCAT-1C test passed: Intentional det(gamma) violation and rollback verified.")

if __name__ == "__main__":
    test_gcat1c_intentional_failure()
    print("Test completed.")