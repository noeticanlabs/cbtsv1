#!/usr/bin/env python3
"""
Noetica (NLLC) Proof-of-Concept Script for GR/NR Integration
Runs 10 steps with audit/rollback and receipts using the GR Host API.
"""

import numpy as np
from src.core.gr_solver import GRSolver
from src.host_api import GRHostAPI
from aeonic_receipts import AeonicReceipts
from receipt_schemas import GRStepReceipt
import hashlib

# Initialize solver
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

# Aeonic receipts for chaining
receipts = AeonicReceipts()
prev_receipt_id = None

# Policy constants (from plan)
H_max = 1e-8
M_max = 1e-8
R_min = -1e10  # Placeholder
dt_min = 1e-8
dt_max = 1e-4
retry_max = 5
dissip_level = 1

print("Starting 10-step NLLC loop with audit/rollback")

for step in range(10):
    print(f"\nStep {step}")

    # 1. Snapshot
    snapshot = host.snapshot()
    state_hash_before = host.get_state_hash()

    # 2. Choose dt (simple policy)
    dt = min(dt_max, max(dt_min, 1e-6))  # Fixed for POC
    print(f"  dt: {dt}")

    # 3. Stage loop (single stage for POC)
    host.step(dt, stage=0)  # UFE step
    host.apply_gauge(dt)   # Gauge

    # 4. Audit
    constraints = host.compute_constraints()
    eps_H, eps_M, R = constraints['eps_H'], constraints['eps_M'], constraints['R']
    metrics = host.energy_metrics()
    H, dH = metrics['H'], metrics['dH']

    print(f"  eps_H: {eps_H}, eps_M: {eps_M}, R: {R}")
    print(f"  H: {H}, dH: {dH}")

    # Gate check
    accept = eps_H <= H_max and eps_M <= M_max
    if not accept:
        print("  REJECT: Constraints exceeded")
        host.reject_step()
        host.restore(snapshot)
        # Retry with dt shrink (simplified, no loop for POC)
        dt *= 0.5
        host.apply_dissipation(dissip_level)
        # Re-try once
        host.step(dt, stage=0)
        host.apply_gauge(dt)
        constraints = host.compute_constraints()
        eps_H, eps_M, R = constraints['eps_H'], constraints['eps_M'], constraints['R']
        metrics = host.energy_metrics()
        H, dH = metrics['H'], metrics['dH']
        accept = eps_H <= H_max and eps_M <= M_max
        if accept:
            print("  RETRY ACCEPT")
        else:
            print("  RETRY REJECT - aborting step")
            continue  # Skip to next

    if accept:
        host.accept_step()
        print("  ACCEPT")

        # Emit receipt
        state_hash_after = host.get_state_hash()
        receipt = GRStepReceipt.create(
            module_id="nllc_gr_poc",
            dep_closure_hash="poc_hash",
            compiler="nllc_poc",
            target="loc-gr-nr",
            step_id=step,
            tau_n=solver.orchestrator.tau,
            dt=dt,
            stage_count=1,
            retry_count=0,  # Simplified
            thread_id="PHY.step.act",
            eps_H=eps_H,
            eps_M=eps_M,
            R=R,
            H=H,
            dH=dH,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            policy_hash=hashlib.sha256(f"{H_max}{M_max}".encode()).hexdigest(),
            prev=prev_receipt_id
        )
        receipts.emit_structured_receipt(receipt)
        prev_receipt_id = receipt.id
        print(f"  Receipt emitted: {receipt.id}")

print("\nProof-of-concept completed. Receipts chained.")