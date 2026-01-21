#!/usr/bin/env python3
"""
Runner for NLLC GR solver test.
Compiles and executes test_comprehensive_gr_solver.nllc using the GR Host API.
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
    # Initialize solver (same as proof-of-concept)
    solver = GRSolver(Nx=16, Ny=16, Nz=16, log_level=20)  # Reduced size for quick test
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
    source_path = "test_comprehensive_gr_solver.nllc"
    with open(source_path, 'r') as f:
        source = f.read()

    # Parse and lower to NIR
    program = parse(source)
    lowerer = Lowerer(source_path)
    module = lowerer.lower_program(program)

    # Create module ID and dep closure hash (simplified)
    module_id = "test_comprehensive_gr_solver"
    dep_closure_hash = hashlib.sha256(b"test_gr_nllc").hexdigest()

    # Create VM with GR host API
    vm = VM(module, module_id, dep_closure_hash, gr_host_api=host)

    # Run the NLLC script
    print("Running NLLC comprehensive GR solver test...")
    result = vm.run()

    # Get receipts
    receipts = vm.get_receipts()
    print(f"Execution completed. Generated {len(receipts)} receipts.")

    # Save receipts
    with open('test_comprehensive_gr_solver_nllc_receipts.json', 'w') as f:
        json.dump(receipts, f, indent=2)

    print("Receipts saved to test_comprehensive_gr_solver_nllc_receipts.json")

if __name__ == "__main__":
    main()