#!/usr/bin/env python3
"""
Phase-1 Validation: Minkowski Stability Test
Run GR solver on flat spacetime, check constraints remain ~0.
"""

import numpy as np
import json
import time
from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import det_sym6, SYM6_IDX

def test_minkowski_stability():
    solver = GRSolver(Nx=10, Ny=10, Nz=10, dx=0.1, dy=0.1, dz=0.1)
    solver.init_minkowski()
    # PhaseLoom boot test: run 10,000 steps, expect stable constraints, dt dominated by phys, zero rollbacks
    solver.run(T_max=10.0)  # Adjust T_max to reach ~10,000 steps with typical dt
    print(f"Final step: {solver.step}, Rollbacks: {solver.orchestrator.rollback_count}")
    # Check constraints remain ~ machine noise
    eps_H_final = solver.constraints.eps_H
    eps_M_final = solver.constraints.eps_M
    print(f"Final eps_H: {eps_H_final:.2e}, eps_M: {eps_M_final:.2e}")
    assert eps_H_final < 1e-10, f"Hamiltonian constraint not stable: {eps_H_final}"
    assert eps_M_final < 1e-10, f"Momentum constraint not stable: {eps_M_final}"
    assert solver.orchestrator.rollback_count == 0, f"Unexpected rollbacks: {solver.orchestrator.rollback_count}"
    print("Minkowski stability test passed.")

def test_phaseloom_minkowski_forever():
    solver = GRSolver(Nx=10, Ny=10, Nz=10, dx=0.1, dy=0.1, dz=0.1)
    solver.init_minkowski()

    # Add increased ripple amplitude constraint-violating seed for testing Loom activation
    solver.fields.gamma_sym6[5,5,5, 0] += 1e-3  # Increased metric perturbation
    solver.fields.K_sym6[5,5,5, 0] += 1e-3      # Increased extrinsic perturbation

    # Run 10,000 steps or until T=100
    solver.run(T_max=100.0)
    
    # Asserts
    det_gamma = det_sym6(solver.fields.gamma_sym6)
    final_det = np.min(det_gamma)
    assert 0.99 < final_det < 1.01, f"detγ drifted to {final_det}"
    
    final_eps_H = solver.constraints.eps_H
    final_eps_M = solver.constraints.eps_M
    assert final_eps_H < 1e-9, f"eps_H > target: {final_eps_H}"
    assert final_eps_M < 1e-9, f"eps_M > target: {final_eps_M}"
    
    # Check dominant clock: all receipts should have "phys" (but loom may activate)
    receipts = solver.orchestrator.receipts.receipts
    dominant_clocks = [r['dominant_clock'] for r in receipts]
    # assert all(c == 'phys' for c in dominant_clocks), f"Non-phys dominance: {set(dominant_clocks)}"
    
    assert solver.orchestrator.rollback_count == 0, f"Rollbacks: {solver.orchestrator.rollback_count}"
    
    print(f"PhaseLoom Minkowski forever: {len(receipts)} steps, detγ={final_det:.6f}, rollbacks=0")

    # Save receipts to JSON
    with open('receipts.json', 'w') as f:
        json.dump(solver.orchestrator.receipts.receipts, f, indent=2)
    print("Receipts saved to receipts.json")

    # Verify phys.margin and risk_gauge
    last_receipt = receipts[-1]
    phys_margin = last_receipt['threads']['phys']['margin']
    risk_gauge = last_receipt['risk_gauge']
    print(f"Final phys.margin: {phys_margin}, risk_gauge: {risk_gauge}")
    assert abs(phys_margin - 0.2) < 1e-6, f"phys.margin not 0.2: {phys_margin}"
    assert abs(risk_gauge - 0.2) < 1e-2, f"risk_gauge not 0.2: {risk_gauge}"

def test_three_kick_gauntlet():
    solver = GRSolver(Nx=10, Ny=10, Nz=10, dx=0.1, dy=0.1, dz=0.1)
    solver.init_minkowski()
    max_steps = 150

    while solver.orchestrator.step < max_steps:
        # Apply kicks before run_step at specific steps
        if solver.orchestrator.step == 9:  # Loom kick at step 10
            epsilon = 1e-4  # small perturbation
            k = 20.0  # high wavenumber
            x = np.arange(solver.fields.Nx) * solver.fields.dx - (solver.fields.Nx * solver.fields.dx) / 2
            y = np.arange(solver.fields.Ny) * solver.fields.dy - (solver.fields.Ny * solver.fields.dy) / 2
            z = np.arange(solver.fields.Nz) * solver.fields.dz - (solver.fields.Nz * solver.fields.dz) / 2
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
            print(f"Applied Loom kick at step {solver.orchestrator.step}")

        elif solver.orchestrator.step == 49:  # Constraint kick at step 50
            offset = 1e-4
            solver.fields.K_sym6 += offset  # increase |K|^2 to offset eps_H/M
            print(f"Applied Constraint kick at step {solver.orchestrator.step}")

        elif solver.orchestrator.step == 99:  # Phys kick at step 100
            boost = 0.5  # temporary increase in alpha to increase v_max
            solver.fields.alpha += boost
            print(f"Applied Phys kick at step {solver.orchestrator.step}")

        dt, dominant_thread, rail_violation = solver.orchestrator.run_step()
        if rail_violation:
            print(f"Stopping due to rail violation at step {solver.orchestrator.step}")
            break

    receipts = solver.orchestrator.receipts.receipts
    print(f"Total steps: {len(receipts)}, Rollbacks: {solver.orchestrator.rollback_count}")
    assert solver.orchestrator.rollback_count == 0, f"Unexpected rollbacks: {solver.orchestrator.rollback_count}"

    # Log receipt excerpts around kicks
    def log_receipts(start, end, label):
        print(f"\n{label} receipts (steps {start}-{end}):")
        for i in range(max(0, start), min(len(receipts), end+1)):
            r = receipts[i]
            print(f"Step {r['step']}: dominant={r['dominant_clock']}, risk_gauge={r['risk_gauge']:.3f}, tight_threads={[t[0] for t in r['tight_threads']]}")

    log_receipts(8, 12, "Loom kick")
    log_receipts(48, 52, "Constraint kick")
    log_receipts(98, 102, "Phys kick")

    print("Three-kick gauntlet test completed.")

def test_adversarial_overlap():
    solver = GRSolver(Nx=10, Ny=10, Nz=10, dx=0.1, dy=0.1, dz=0.1)
    solver.init_minkowski()
    
    # Explicitly zero Z fields to ensure stability with damping
    solver.fields.Z = np.zeros((10, 10, 10))
    solver.fields.Z_i = np.zeros((10, 10, 10, 3))

    # Adjust rails and eps_max for adversarial test
    solver.orchestrator.rails.H_max = 1e-2
    solver.orchestrator.rails.M_max = 1e-3
    solver.orchestrator.threads.eps_H_max = 1e-7
    solver.orchestrator.threads.eps_M_max = 1e-3
    solver.stepper.lambda_val = 0.1  # Enable damping
    
    # Force smaller dt for stability
    solver.scheduler.compute_dt = lambda eps_H, eps_M: 0.01
    
    max_steps = 15  # Reach step 10+ for kicks

    while solver.orchestrator.step < max_steps:
        # Apply both kicks simultaneously at step 9 (before step 10)
        if solver.orchestrator.step == 9:
            # Loom kick: high wavenumber ripple in metric
            epsilon = 1e-2
            k = 20.0
            x = np.arange(solver.fields.Nx) * solver.fields.dx - (solver.fields.Nx * solver.fields.dx) / 2
            y = np.arange(solver.fields.Ny) * solver.fields.dy - (solver.fields.Ny * solver.fields.dy) / 2
            z = np.arange(solver.fields.Nz) * solver.fields.dz - (solver.fields.Nz * solver.fields.dz) / 2
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            solver.fields.gamma_sym6[..., SYM6_IDX["xx"]] += epsilon * np.sin(k * X) * np.sin(k * Y) * np.sin(k * Z)
            print(f"Applied Loom kick at step {solver.orchestrator.step}")

            # Constraint kick: increase |K|^2 to violate constraints
            offset = 1e-3
            solver.fields.K_sym6 += offset
            print(f"Applied Constraint kick at step {solver.orchestrator.step}")

        dt, dominant_thread, rail_violation = solver.orchestrator.run_step()
        if rail_violation:
            print(f"Stopping due to rail violation at step {solver.orchestrator.step}")
            break

    receipts = solver.orchestrator.receipts.receipts
    print(f"Total steps: {len(receipts)}, Rollbacks: {solver.orchestrator.rollback_count}")
    assert solver.orchestrator.rollback_count == 0, f"Unexpected rollbacks: {solver.orchestrator.rollback_count}"

    # Retrieve receipt for step 10
    receipt_10 = next((r for r in receipts if r['step'] == 10), None)
    if receipt_10 is None:
        raise AssertionError("No receipt found for step 10")

    print(f"Receipt 10: {receipt_10}")

    # Verify dt = min of all non-None dts from threads and dt_loom
    dts = []
    loom_cap = receipt_10.get('dt_loom')
    if loom_cap is not None and np.isfinite(loom_cap):
        dts.append(loom_cap)
    for thread_data in receipt_10['threads'].values():
        if thread_data['dt'] is not None and np.isfinite(thread_data['dt']):
            dts.append(thread_data['dt'])
    if not dts:
        raise AssertionError("No valid dts found")
    rho_target = solver.orchestrator.threads.rho_target  # 0.8
    dt_min = rho_target * min(dts)
    assert receipt_10['dt'] <= dt_min + 1e-12, f"dt {receipt_10['dt']} > rho*min(dts) {dts} = {dt_min}"

    # Verify dominance = argmin(margin) among all threads with dt not None and loom if dt_loom not None
    thread_margins = {k: p['margin'] for k, p in receipt_10['threads'].items() if p['dt'] is not None}
    dt_loom = receipt_10.get('dt_loom')
    if dt_loom is not None and dt_loom > 0:
        loom_margin = 1 - (receipt_10['dt'] / dt_loom) if dt_loom > 0 else 1.0
        thread_margins['loom'] = loom_margin
    # Find winner (min margin, since smaller margin means more constraining)
    dominance_expected = min(thread_margins, key=thread_margins.get)
    assert receipt_10['dominant_clock'] == dominance_expected, f"Dominance {receipt_10['dominant_clock']} != expected {dominance_expected}"

    # Verify loser shows small margin gap
    if len(thread_margins) == 2:
        margins_list = list(thread_margins.values())
        gap = abs(margins_list[0] - margins_list[1])
        assert gap < 0.01, f"Margin gap {gap} not small enough"
        print(f"Adversarial overlap: dt={receipt_10['dt']:.6f}, dominance={receipt_10['dominant_clock']}, margin_gap={gap:.6f}")
    else:
        print("Only one active thread, skipping gap check")

    # Save receipt for verification
    with open('receipts_adversarial.json', 'w') as f:
        json.dump(receipt_10, f, indent=2)

    print("Adversarial overlap test passed.")

if __name__ == "__main__":
    test_minkowski_stability()