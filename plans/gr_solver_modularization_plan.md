# Plan for Modularizing `gr_solver/gr_stepper.py`

## 1. Executive Summary

The current `gr_solver/gr_stepper.py` is a monolithic file that handles multiple distinct responsibilities: time integration (RK4), RHS physics computation, audit receipt generation, time-step scheduling, and step validation (gates). This violates the single-responsibility principle and makes the code difficult to maintain, debug, and test.

This plan outlines a refactoring strategy to break down `GRStepper` into smaller, more focused components, aligning with the architecture specified in `plans/gr_solver_build_plan.md`. We will introduce the following new or updated modules:

*   **`gr_rhs.py`**: To encapsulate the physics of the RHS computation.
*   **`gr_ledger.py`**: To manage the creation and emission of all Ω-receipts.
*   **`gr_scheduler.py`**: To handle adaptive time-step (`dt`) calculation.
*   **`gr_gatekeeper.py`**: To centralize the logic for step acceptance, rejection, and corrective actions.

The `gr_stepper.py` module will be refactored to act as a high-level orchestrator for the UFE time-stepping process, delegating tasks to these new, specialized components.

## 2. Assessment of `gr_stepper.py`

The `GRStepper` class currently has the following responsibilities mixed together:

*   **Time Integration**: The `step_ufe` method contains the main RK4 loop logic.
*   **Physics RHS Computation**: The `compute_rhs` method and its JIT-compiled helper (`_compute_gamma_tilde_rhs_jit`) contain the complex physics equations for the BSSN and ADM formulations.
*   **Auditing & Receipts**: Numerous methods (`emit_stage_rhs_receipt`, `emit_clock_decision_receipt`, `emit_ledger_eval_receipt`, `emit_step_receipt`) are dedicated to generating and writing a detailed audit trail. This is the "Ledger" function.
*   **Timestep Calculation**: The `compute_clocks` method determines the appropriate `dt` based on various physical constraints (CFL, gauge, etc.). This is a "Scheduler" function.
*   **Step Validation**: `check_gates` and `check_gates_internal` are responsible for validating the results of a time step against a policy (`rails_policy`).
*   **Corrective Actions**: `apply_corrections` and `apply_damping` implement logic to respond to constraint violations or instabilities.

## 3. Proposed Modular Architecture

I will refactor the codebase into the following components:

| Module | Class | Responsibility | Methods to be Moved from `GRStepper` |
| :--- | :--- | :--- | :--- |
| **`gr_rhs.py`** | `GRRhs` | Computes the right-hand-side (RHS) of the evolution equations. | `compute_rhs`, `_compute_gamma_tilde_rhs_jit` |
| **`gr_ledger.py`** | `GRLedger` | Manages the creation, hashing, and writing of all Ω-receipts for the audit trail. | `emit_stage_rhs_receipt`, `emit_clock_decision_receipt`, `emit_ledger_eval_receipt`, `emit_step_receipt`, and helper methods for hashing/chaining. |
| **`gr_scheduler.py`**| `GRScheduler`| Calculates the optimal time-step (`dt`) based on physical and numerical constraints. | `compute_clocks` |
| **`gr_gatekeeper.py`**| `GRGatekeeper`| Enforces step validation policies (`rails`), checks for violations, and applies corrective actions. | `check_gates`, `check_gates_internal`, `apply_corrections`, `apply_damping` |
| **`gr_stepper.py`** | `GRStepper` | Orchestrates the RK4 time-stepping process by coordinating the other components. | (Will retain `step`, `step_ufe`) |

## 4. Detailed Refactoring Plan

**Step 1: Create `gr_rhs.py` - The RHS Computer**

1.  Create a new file: `gr_solver/gr_rhs.py`.
2.  Create a class `GRRhs` inside it.
3.  Move the `compute_rhs` method and the static JIT helper `_compute_gamma_tilde_rhs_jit` from `gr_stepper.py` into the `GRRhs` class.
4.  The `GRRhs` constructor will take `fields` and `geometry` objects as dependencies.
5.  The `GRStepper` will instantiate `self.rhs_computer = GRRhs(self.fields, self.geometry)` in its constructor.
6.  In `step_ufe`, calls to `self.compute_rhs(...)` will be replaced with `self.rhs_computer.compute_rhs(...)`.

**Step 2: Create `gr_ledger.py` - The Auditor**

1.  Create a new file: `gr_solver/gr_ledger.py`.
2.  Create a class `GRLedger`. Its constructor will manage the `receipts_file` and `prev_receipt_hash`.
3.  Move all receipt-related methods (`emit_stage_rhs_receipt`, `emit_clock_decision_receipt`, `emit_step_receipt`, etc.) from `gr_stepper.py` into `GRLedger`.
4.  The `GRStepper` will instantiate `self.ledger = GRLedger()` in its `__init__`.
5.  All calls to `self.emit_*_receipt(...)` in `step_ufe` will be replaced with `self.ledger.emit_*_receipt(...)`, passing in the necessary data (fields, step number, time, etc.).

**Step 3: Create `gr_scheduler.py` - The Time Keeper**

1.  Create a new file: `gr_solver/gr_scheduler.py`.
2.  Create a class `GRScheduler`.
3.  Move the `compute_clocks` method from `gr_stepper.py` to `GRScheduler`.
4.  The `GRStepper` will instantiate `self.scheduler = GRScheduler(self.fields)` in its `__init__`.
5.  The call to `self.compute_clocks(dt)` in `step_ufe` will become `self.scheduler.compute_clocks(dt)`.

**Step 4: Create `gr_gatekeeper.py` - The Validator**

1.  Create a new file: `gr_solver/gr_gatekeeper.py`.
2.  Create a class `GRGatekeeper`. Its constructor will take `fields`, `constraints`, and the `loc_operator` as dependencies.
3.  Move `check_gates`, `check_gates_internal`, `apply_corrections`, and `apply_damping` from `gr_stepper.py` into this new class.
4.  The `GRStepper` will instantiate `self.gatekeeper = GRGatekeeper(...)` in its `__init__`.
5.  Calls within `step_ufe` will be updated:
    *   `self.check_gates_internal(...)` -> `self.gatekeeper.check_gates(...)`
    *   `self.apply_corrections(...)` -> `self.gatekeeper.apply_corrections(...)`
    *   `self.apply_damping()` -> `self.gatekeeper.apply_damping()`

**Step 5: Refactor `gr_stepper.py` - The Orchestrator**

1.  The `GRStepper`'s `__init__` method will be simplified to instantiate the new components (`GRRhs`, `GRLedger`, `GRScheduler`, `GRGatekeeper`). It will pass the necessary dependencies (like `fields`, `geometry`, `constraints`) to them.
2.  The `step_ufe` method will be streamlined. It will maintain the high-level RK4 control flow but will delegate all major tasks to the new components.

## 5. Benefits of This Refactoring

*   **Separation of Concerns**: Each module will have a single, well-defined responsibility.
*   **Improved Readability**: `gr_stepper.py` will become a concise, high-level description of the time-stepping algorithm.
*   **Enhanced Maintainability**: Changes to the physics equations (`gr_rhs.py`) will not risk breaking the audit trail logic (`gr_ledger.py`).
*   **Easier Testing**: Each component can be unit-tested in isolation, simplifying bug identification.
*   **Alignment with Project Goals**: This plan directly implements the modular architecture envisioned in `plans/gr_solver_build_plan.md`.

## 6. Risks & Mitigations

- **Risk**: The refactoring process could introduce new bugs into a critical, complex component.
- **Mitigation**: The proposed modular structure is designed for testability. Each new module (`GRLedger`, `GRRhs`, `GRScheduler`, `GRGatekeeper`) will have its own suite of unit tests to verify its functionality in isolation. The existing test suite for `gr_stepper` will be adapted to test the integrated orchestrator, ensuring no regressions in behavior.

## 7. Conclusion

This plan provides a systematic approach to refactoring the system's core stepper logic to enhance modularity, reliability, and maintainability. By breaking down the monolithic `GRStepper` into smaller, well-defined components, we can improve the long-term health and velocity of the project.
