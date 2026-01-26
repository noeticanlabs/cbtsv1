#!/usr/bin/env python3
"""
Runner for WP1 Global Smoothness Bootstrap NLLC test.
Compiles and executes tests/wp1_global_smoothness_bootstrap.nllc using the GR Host API.
"""

import json
import sys
import os
sys.path.append('src')

from gr_solver.gr_solver import GRSolver
from gr_solver.host_api import GRHostAPI
from nllc.vm import VM
from nllc.parse import parse
from nllc.lower_nir import Lowerer
import hashlib

def main():
    print("=" * 70)
    print("WP1 Global Smoothness Bootstrap Test (NLLC Runner)")
    print("=" * 70)
    
    # Initialize solver (reduced size for quick test)
    solver = GRSolver(Nx=16, Ny=16, Nz=16, log_level=20)
    solver.init_minkowski()

    # Create host API
    host = GRHostAPI(
        fields=solver.fields,
        geometry=solver.geometry,
        constraints=solver.constraints,
        gauge=solver.gauge,
        stepper=solver.stepper,
        orchestrator=solver.orchestrator
    )

    # Compile NLLC script
    source_path = "tests/wp1_global_smoothness_bootstrap.nllc"
    if not os.path.exists(source_path):
        print(f"ERROR: NLLC test file not found at {source_path}")
        return 1
        
    with open(source_path, 'r') as f:
        source = f.read()

    # Parse and lower to NIR
    program = parse(source)
    lowerer = Lowerer(source_path)
    module = lowerer.lower_program(program)

    # Create module ID and dep closure hash
    module_id = "wp1_global_smoothness_bootstrap"
    dep_closure_hash = hashlib.sha256(b"wp1_bootstrap_nllc").hexdigest()

    # Create VM with GR host API
    vm = VM(module, module_id, dep_closure_hash, gr_host_api=host)

    # Run the NLLC script
    print("Running WP1 Global Smoothness Bootstrap NLLC test...")
    try:
        result = vm.run()
    except Exception as e:
        print(f"Test execution encountered an error: {e}")
        # Still try to save receipts even if there was an error
        result = None

    # Get receipts
    receipts = vm.get_receipts()
    print(f"Execution completed. Generated {len(receipts)} receipts.")

    # Save receipts
    receipts_file = 'wp1_bootstrap_nllc_receipts.json'
    with open(receipts_file, 'w') as f:
        json.dump(receipts, f, indent=2)
    print(f"Receipts saved to {receipts_file}")

    # Check if certificate was generated in receipts
    certificate = None
    for receipt in receipts:
        if receipt.get('type') == 'certificate' or 'certificate' in receipt:
            certificate = receipt.get('certificate') or receipt
            break
    
    if certificate:
        cert_file = 'wp1_bootstrap_certificate.json'
        with open(cert_file, 'w') as f:
            json.dump(certificate, f, indent=2)
        print(f"Certificate saved to {cert_file}")
        print(f"\nCertificate Summary:")
        print(f"  - bootstrap_passed: {certificate.get('summary', {}).get('bootstrap_passed', 'N/A')}")
        print(f"  - max_energy: {certificate.get('energy_boundedness', {}).get('max', 'N/A')}")
        print(f"  - max_eps_H: {certificate.get('constraint_coherence', {}).get('max_hamiltonian', 'N/A')}")
        print(f"  - max_eps_M: {certificate.get('constraint_coherence', {}).get('max_momentum', 'N/A')}")
        print(f"  - total_rail_actions: {certificate.get('rail_spending', {}).get('total_actions', 'N/A')}")
    else:
        print("NOTE: No certificate found in receipts (host function not fully implemented)")

    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
