import numpy as np
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from gr_solver.gr_core_fields import GRCoreFields, SYM6_IDX, det_sym6
from gr_solver.gr_geometry import GRGeometry
from gr_solver.gr_constraints import GRConstraints

# NSC imports
from src.nllc import parse, lower_nir
from src.nllc.vm import VM

# Load coupling policy
with open('coupling_policy_v0.1.json', 'r') as f:
    coupling_policy = json.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRHostAPI:
    """Host API for GR operations in NSC VM."""
    def __init__(self, constraints):
        self.constraints = constraints

    def compute_constraints(self):
        """Compute and return constraint residuals."""
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()
        self.constraints.compute_residuals()
        return {'eps_H': float(self.constraints.eps_H), 'eps_M': float(self.constraints.eps_M)}

class SchwarzschildNSCExactTest:
    """
    NSC-based test: Initialize Schwarzschild metric in Python track,
    then execute constraint check in NSC VM track using coupling.
    """

    def __init__(self, N=32, L=10.0, M=1.0):
        self.N = N
        self.L = L
        self.M = M
        self.dx = L / N
        self.fields = GRCoreFields(N, N, N, self.dx, self.dx, self.dx)
        self.geometry = GRGeometry(self.fields)
        self.constraints = GRConstraints(self.fields, self.geometry)

    def init_schwarzschild(self):
        """Initialize fields to Schwarzschild metric in isotropic coordinates."""
        logger.info("Initializing Schwarzschild metric...")

        # Create centered coordinates
        x = (np.arange(self.N) - self.N//2) * self.dx
        y = (np.arange(self.N) - self.N//2) * self.dx
        z = (np.arange(self.N) - self.N//2) * self.dx
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        r = np.sqrt(X**2 + Y**2 + Z**2)
        # Avoid division by zero at r=0, and large values near center
        r = np.maximum(r, 0.1)

        # Conformal factor ψ
        psi = np.sqrt(1.0 + self.M / (2.0 * r))

        # Spatial metric γ_ij = ψ^4 δ_ij
        gamma_factor = psi**4
        self.fields.gamma_sym6[..., SYM6_IDX["xx"]] = gamma_factor
        self.fields.gamma_sym6[..., SYM6_IDX["yy"]] = gamma_factor
        self.fields.gamma_sym6[..., SYM6_IDX["zz"]] = gamma_factor
        self.fields.gamma_sym6[..., SYM6_IDX["xy"]] = 0.0
        self.fields.gamma_sym6[..., SYM6_IDX["xz"]] = 0.0
        self.fields.gamma_sym6[..., SYM6_IDX["yz"]] = 0.0

        # Lapse α
        alpha_num = 1.0 - self.M / (2.0 * r)
        alpha_den = 1.0 + self.M / (2.0 * r)
        self.fields.alpha = alpha_num / alpha_den

        # Shift β^i = 0
        self.fields.beta.fill(0.0)

        # Extrinsic curvature K_ij = 0
        self.fields.K_sym6.fill(0.0)

        # BSSN fields
        # Conformal factor φ = ln(ψ)
        self.fields.phi = np.log(psi)
        # Conformal metric γ̃_ij = δ_ij (since γ_ij = ψ^4 γ̃_ij)
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["xx"]] = 1.0
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["yy"]] = 1.0
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["zz"]] = 1.0
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["xy"]] = 0.0
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["xz"]] = 0.0
        self.fields.gamma_tilde_sym6[..., SYM6_IDX["yz"]] = 0.0
        # Trace-free A_ij = 0
        self.fields.A_sym6.fill(0.0)
        # Gamma_tilde^i = 0
        self.fields.Gamma_tilde.fill(0.0)

        # Active fields
        self.fields.Z.fill(0.0)
        self.fields.Z_i.fill(0.0)

        logger.info("Schwarzschild initialization complete.")

    def run_nsc_test(self):
        """Run the NSC-based constraint check."""
        logger.info("Running NSC-based Schwarzschild Exact Solution Test...")

        # Compute geometry
        self.geometry.compute_christoffels()
        self.geometry.compute_ricci()
        self.geometry.compute_scalar_curvature()

        # Load and parse NLLC script
        nllc_path = os.path.join(os.path.dirname(__file__), 'test_schwarzschild_nsc.nllc')
        with open(nllc_path, 'r') as f:
            nllc_source = f.read()

        ast = parse.parse(nllc_source)
        lowerer = lower_nir.Lowerer('test_schwarzschild_nsc.nllc')
        nir_module = lowerer.lower_program(ast)

        # Create GR host API
        host_api = GRHostAPI(self.constraints)

        # Create VM with dummy module_id and dep_closure_hash
        vm = VM(nir_module, module_id='test_schwarzschild_nsc', dep_closure_hash='dummy_hash', gr_host_api=host_api)

        # Run VM
        success = vm.run()

        # Get receipts
        receipts = vm.get_receipts()

        logger.info(f"NSC test result: {success}")

        # Save results
        results = {
            'passed': bool(success),
            'N': self.N,
            'L': self.L,
            'M': self.M,
            'receipts': receipts
        }

        with open('test_schwarzschild_nsc_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to test_schwarzschild_nsc_results.json")

        return success

    def run(self):
        """Full test: init and run NSC check."""
        self.init_schwarzschild()
        return self.run_nsc_test()

if __name__ == "__main__":
    test = SchwarzschildNSCExactTest()
    passed = test.run()
    print(f"Schwarzschild NSC Test Passed: {passed}")