import numpy as np
import sys
import os
import json
import logging
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import SYM6_IDX, det_sym6
from aeonic_memory_bank import AeonicMemoryBank

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveGRSolverTest:
    """
    Comprehensive system function test for GR solver.
    Exercises all major functions: memory, rails, gates, geometry, constraints, stepping.
    """

    def __init__(self, N=8, L=4.0, dt=None):
        self.N = N
        self.Nx = self.Ny = self.Nz = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.solver = GRSolver(N, N, N, self.dx, self.dx, self.dx)
        self.results = {}
        logger.info(f"Initialized test with grid {N}x{N}x{N}, L={L}, dt={dt}")

    def test_initialization(self):
        """Test Minkowski initialization and initial computations."""
        logger.info("Testing initialization...")
        try:
            self.solver.init_minkowski()

            # Check fields are finite
            assert np.all(np.isfinite(self.solver.fields.gamma_sym6)), "gamma_sym6 not finite"
            assert np.all(np.isfinite(self.solver.fields.K_sym6)), "K_sym6 not finite"
            assert np.all(np.isfinite(self.solver.fields.alpha)), "alpha not finite"
            assert np.all(self.solver.fields.alpha > 0), "alpha not positive"

            # Check geometry computations
            assert hasattr(self.solver.geometry, 'Gamma'), "Christoffels not computed"
            assert hasattr(self.solver.geometry, 'ricci'), "Ricci not computed"
            assert self.solver.geometry.ricci is not None, "Ricci is None"

            # Check constraints
            assert np.isfinite(self.solver.constraints.eps_H), "eps_H not finite"
            assert np.isfinite(self.solver.constraints.eps_M), "eps_M not finite"

            # Check det(gamma) > 0
            det_gamma = det_sym6(self.solver.fields.gamma_sym6)
            assert np.all(det_gamma > 0), "det(gamma) not positive"

            self.results['initialization'] = {
                'passed': True,
                'eps_H': float(self.solver.constraints.eps_H),
                'eps_M': float(self.solver.constraints.eps_M),
                'det_gamma_min': float(np.min(det_gamma))
            }
            logger.info("Initialization test passed")
            return True

        except Exception as e:
            logger.error(f"Initialization test failed: {e}")
            self.results['initialization'] = {'passed': False, 'error': str(e)}
            return False

    def test_memory_operations(self):
        """Test AeonicMemoryBank operations."""
        logger.info("Testing memory operations...")
        try:
            memory = self.solver.memory_bank

            # Test put operation
            test_data = np.ones((10, 10))
            bytes_est = test_data.nbytes
            memory.put("test_key", 1, test_data, bytes_est, ttl_s=1000, ttl_l=10000,
                      recompute_cost_est=1.0, risk_score=0.1, tainted=False, regime_hashes=[])

            # Test get operation
            retrieved = memory.get("test_key")
            assert retrieved is not None, "Failed to retrieve test data"
            assert np.array_equal(retrieved, test_data), "Retrieved data mismatch"

            # Test maintenance tick
            memory.maintenance_tick()

            self.results['memory_operations'] = {'passed': True}
            logger.info("Memory operations test passed")
            return True

        except Exception as e:
            logger.error(f"Memory operations test failed: {e}")
            self.results['memory_operations'] = {'passed': False, 'error': str(e)}
            return False

    def test_single_step(self):
        """Test single orchestrator step."""
        logger.info("Testing single step...")
        try:
            dt, dominant_thread, rail_violation = self.solver.orchestrator.run_step(dt_max=self.dt)

            assert np.isfinite(dt), "dt not finite"
            assert isinstance(dominant_thread, str), "dominant_thread not string"
            assert rail_violation is None or isinstance(rail_violation, str), "rail_violation invalid"

            # Check fields still finite after step
            assert np.all(np.isfinite(self.solver.fields.gamma_sym6)), "gamma_sym6 not finite after step"
            assert np.all(np.isfinite(self.solver.fields.alpha)), "alpha not finite after step"

            self.results['single_step'] = {
                'passed': True,
                'dt': float(dt),
                'dominant_thread': dominant_thread,
                'rail_violation': rail_violation,
                'step': self.solver.orchestrator.step,
                't': float(self.solver.orchestrator.t)
            }
            logger.info("Single step test passed")
            return True

        except Exception as e:
            import traceback
            logger.error(f"Single step test failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.results['single_step'] = {'passed': False, 'error': str(e), 'traceback': traceback.format_exc()}
            return False

    def test_multi_step_evolution(self):
        """Test multi-step evolution."""
        logger.info("Testing multi-step evolution...")
        try:
            num_steps = 10
            eps_H_history = []
            eps_M_history = []
            dominant_threads = []
            t_history = []

            for i in range(num_steps):
                dt, dominant_thread, rail_violation = self.solver.orchestrator.run_step(dt_max=self.dt)
                eps_H_history.append(float(self.solver.constraints.eps_H))
                eps_M_history.append(float(self.solver.constraints.eps_M))
                dominant_threads.append(dominant_thread)
                t_history.append(float(self.solver.orchestrator.t))

                # Check no catastrophic failures
                assert np.isfinite(dt), f"dt not finite at step {i}"
                assert np.all(np.isfinite(self.solver.fields.gamma_sym6)), f"gamma_sym6 not finite at step {i}"

            # Collect stage_eps_H from receipts
            receipts = self.solver.orchestrator.receipts.receipts
            stage_eps_H_history = [r.get('stage_eps_H', {}) for r in receipts[-num_steps:] if 'stage_eps_H' in r]

            self.results['multi_step_evolution'] = {
                'passed': True,
                'num_steps': num_steps,
                'eps_H_history': eps_H_history,
                'eps_M_history': eps_M_history,
                'dominant_threads': dominant_threads,
                't_history': t_history,
                'stage_eps_H_history': stage_eps_H_history,
                'final_eps_H': eps_H_history[-1],
                'final_eps_M': eps_M_history[-1],
                'final_t': t_history[-1]
            }
            logger.info("Multi-step evolution test passed")
            return True

        except Exception as e:
            logger.error(f"Multi-step evolution test failed: {e}")
            self.results['multi_step_evolution'] = {'passed': False, 'error': str(e)}
            return False

    def test_rails_gates(self):
        """Test rails and gates functionality."""
        logger.info("Testing rails and gates...")
        try:
            rails = self.solver.orchestrator.rails

            # Check gates
            eps_H, eps_M, sem_ok, sem_reason = self.solver.orchestrator.sem_safe_compute_residuals()
            gate_violation = rails.check_gates(eps_H, eps_M, self.solver.geometry, self.solver.fields)

            # Compute margins
            margins = rails.compute_margins(eps_H, eps_M, self.solver.geometry, self.solver.fields, self.solver.orchestrator.threads.m_det_min)

            self.results['rails_gates'] = {
                'passed': True,
                'gate_violation': gate_violation,
                'margins': {k: float(v) for k, v in margins.items()},
                'sem_ok': sem_ok,
                'sem_reason': sem_reason
            }
            logger.info("Rails and gates test passed")
            return True

        except Exception as e:
            logger.error(f"Rails and gates test failed: {e}")
            self.results['rails_gates'] = {'passed': False, 'error': str(e)}
            return False

    def test_phaseloom_threads(self):
        """Test PhaseLoom thread computations."""
        logger.info("Testing PhaseLoom threads...")
        try:
            phaseloom = self.solver.phaseloom

            # Compute proxy residuals (placeholder test)
            residuals = phaseloom.compute_residuals(self.solver.fields, self.solver.geometry, self.solver.gauge)

            # Test dt arbitration
            dt_min, dominant = phaseloom.arbitrate_dt(residuals)

            self.results['phaseloom_threads'] = {
                'passed': True,
                'num_residuals': len(residuals),
                'dt_min': float(dt_min),
                'dominant_thread': dominant
            }
            logger.info("PhaseLoom threads test passed")
            return True

        except Exception as e:
            logger.error(f"PhaseLoom threads test failed: {e}")
            self.results['phaseloom_threads'] = {'passed': False, 'error': str(e)}
            return False

    def test_constraints_geometry(self):
        """Test constraints and geometry computations."""
        logger.info("Testing constraints and geometry...")
        try:
            # Geometry checks
            gamma_finite = np.all(np.isfinite(self.solver.geometry.Gamma))
            assert gamma_finite, "Christoffels not finite"
            if self.solver.geometry.ricci is not None:
                ricci_finite = np.all(np.isfinite(self.solver.geometry.ricci))
                assert ricci_finite, "Ricci not finite"
            if hasattr(self.solver.geometry.R, '__len__') and len(self.solver.geometry.R) > 1:
                R_val = float(np.max(np.abs(self.solver.geometry.R)))
            else:
                R_val = float(self.solver.geometry.R)
            assert np.isfinite(R_val), "Scalar curvature not finite"

            # Constraints checks
            eps_H_val = float(self.solver.constraints.eps_H)
            eps_M_val = float(self.solver.constraints.eps_M)
            assert np.isfinite(eps_H_val), "eps_H not finite"
            assert np.isfinite(eps_M_val), "eps_M not finite"

            # Check bounded (not exploded)
            assert abs(eps_H_val) < 1e6, "eps_H too large"
            assert abs(eps_M_val) < 1e6, "eps_M too large"

            self.results['constraints_geometry'] = {
                'passed': True,
                'eps_H': eps_H_val,
                'eps_M': eps_M_val,
                'R': R_val
            }
            logger.info("Constraints and geometry test passed")
            return True

        except Exception as e:
            logger.error(f"Constraints and geometry test failed: {e}")
            self.results['constraints_geometry'] = {'passed': False, 'error': str(e)}
            return False

    def run_all_tests(self, output_file=None):
        """Run all test methods and collect results."""
        logger.info("Running comprehensive GR solver system tests...")

        test_methods = [
            self.test_initialization,
            self.test_memory_operations,
            self.test_single_step,
            self.test_multi_step_evolution,
            self.test_rails_gates,
            self.test_phaseloom_threads,
            self.test_constraints_geometry
        ]

        passed = 0
        total = len(test_methods)

        for test_method in test_methods:
            if test_method():
                passed += 1

        overall_passed = passed == total

        # Get last state if available
        last_state = getattr(self.solver.orchestrator, 'last_state', None)
        summary_extra = {}
        if last_state:
            summary_extra.update({
                'dx': float(self.dx),
                'N': self.N,  # Assuming cubic
                'L': float(self.L),
                'dt_target': float(self.dt) if self.dt else None,
                'dt_commit': float(last_state.dt_commit) if last_state.dt_commit else None,
                'dt_ratio': float(last_state.dt_ratio) if last_state.dt_ratio else None,
                'n_substeps': last_state.n_substeps,
                'substep_cap_hit': last_state.substep_cap_hit,
                'eps_H_gate': float(last_state.eps_H_gate) if last_state.eps_H_gate else None,
                'eps_H_factor': float(last_state.eps_H_factor) if last_state.eps_H_factor else None,
                'dominance_note': last_state.dominance_note
            })

        # Add final t from multi_step if available
        if 'multi_step_evolution' in self.results and self.results['multi_step_evolution']['passed']:
            summary_extra['final_t'] = self.results['multi_step_evolution']['final_t']

        self.results['summary'] = {
            'overall_passed': overall_passed,
            'tests_passed': passed,
            'total_tests': total,
            'grid': f"{self.Nx}x{self.Ny}x{self.Nz}",
            'L': self.L,
            **summary_extra
        }

        # Save results
        if output_file is None:
            output_file = f'test_E1_N{self.N}.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Tests completed: {passed}/{total} passed")
        return overall_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive GR solver test.')
    parser.add_argument('--N', type=int, default=8, help='Grid size N')
    parser.add_argument('--dt', type=float, required=True, help='Time step dt')
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()
    test = ComprehensiveGRSolverTest(N=args.N, L=4.0, dt=args.dt)
    passed = test.run_all_tests(output_file=args.output)
    print(f"Comprehensive GR Solver Test Passed: {passed}")