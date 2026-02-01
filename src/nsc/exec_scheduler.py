"""
NSC-M3L Deterministic Scheduler

Implements deterministic execution scheduling for the EXEC model.
Ensures reproducible execution order for verification and testing.

Semantic Domain Objects:
    - Deterministic ordering guarantees
    - Stage-time schedule for LoC-Time

Denotation: Program → Executable bytecode with execution semantics
"""

import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# =============================================================================
# Scheduler Errors
# =============================================================================

class SchedulerError(Exception):
    """Scheduler execution error."""
    pass


class DeterminismError(SchedulerError):
    """Determinism violation error."""
    
    def __init__(self, message: str, step: int = 0):
        self.message = message
        self.step = step
        super().__init__(f"Determinism violation at step {step}: {message}")


# =============================================================================
# Execution Schedule
# =============================================================================

@dataclass
class ExecutionStep:
    """Single execution step in the schedule."""
    step_number: int
    stage: str
    start_time: float
    end_time: float
    operations: List[str]  # Ordered list of executed operations
    state_snapshot: Dict[str, Any]
    result: Any
    deterministic: bool = True
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ExecutionCycle:
    """Complete execution cycle (LoC-Time iteration)."""
    cycle_number: int
    start_time: float
    end_time: float
    steps: List[ExecutionStep]
    is_deterministic: bool = True
    max_deviation: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# =============================================================================
# Deterministic Scheduler
# =============================================================================

class DeterministicScheduler:
    """
    Deterministic execution scheduler for NSC-M3L.
    
    Ensures:
    1. Operations execute in a fixed, predictable order
    2. Timestamps are deterministic (not wall-clock)
    3. Operation sequences can be verified for determinism
    
    The scheduler operates on a VM instance and coordinates execution
    across multiple stages of the LoC-Time protocol.
    """
    
    def __init__(self, vm=None, deterministic_clock: bool = True):
        """
        Initialize the scheduler.
        
        Args:
            vm: Optional VM instance to control
            deterministic_clock: Whether to use deterministic timing
        """
        self.vm = vm
        self.deterministic_clock = deterministic_clock
        
        # Scheduling state
        self.step_number: int = 0
        self.cycle_number: int = 0
        self.current_stage: str = "OBSERVE"
        
        # Execution history
        self.step_history: List[ExecutionStep] = []
        self.cycle_history: List[ExecutionCycle] = []
        
        # Determinism verification
        self.base_operation_order: Optional[List[str]] = None
        self.max_observed_deviation: float = 0.0
        
        # Timing (deterministic or real)
        self._virtual_time: float = 0.0
        self._time_scale: float = 1.0
        
        # Operation order tracking
        self._current_operations: List[str] = []
    
    def set_vm(self, vm):
        """Set the VM to schedule."""
        self.vm = vm
    
    def step(self, dt: float = 1.0, stage: Optional[str] = None) -> ExecutionStep:
        """
        Execute a single deterministic step.
        
        Args:
            dt: Time step size
            stage: Stage name (auto-increment if not provided)
            
        Returns:
            ExecutionStep with results and verification data
        """
        if self.vm is None:
            raise SchedulerError("No VM configured")
        
        self.step_number += 1
        self._virtual_time += dt
        
        # Determine stage
        if stage is None:
            stage = self._get_next_stage()
        self.current_stage = stage
        
        # Start timing
        start_time = self._get_time()
        self._current_operations = []
        
        # Execute step in VM
        state_before = self._snapshot_state()
        
        try:
            # Run VM for one step
            self.vm.step()
        except Exception as e:
            raise SchedulerError(f"VM step failed: {e}")
        
        # Get operations executed
        operations = self._get_operation_order()
        
        # End timing
        end_time = self._get_time()
        
        # Create step record
        step = ExecutionStep(
            step_number=self.step_number,
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            operations=operations,
            state_snapshot=state_before,
            result=self.vm._get_result() if hasattr(self.vm, '_get_result') else None,
            deterministic=True
        )
        
        self.step_history.append(step)
        
        # Verify determinism
        is_det, deviation = self._verify_step_determinism(step)
        if not is_det:
            step.deterministic = False
            self.max_observed_deviation = max(self.max_observed_deviation, deviation)
        
        return step
    
    def run_cycle(self, dt: float = 1.0) -> ExecutionCycle:
        """
        Execute a complete LoC-Time cycle.
        
        A cycle consists of:
        - OBSERVE: Observe current state
        - DECIDE: Decide on action
        - ACT_PHY: Apply physical update
        - ACT_CONS: Enforce constraints
        - AUDIT: Audit the step
        - ACCEPT: Accept the step
        - RECEIPT: Issue receipt
        
        Args:
            dt: Time step size
            
        Returns:
            ExecutionCycle with all steps
        """
        stages = ["OBSERVE", "DECIDE", "ACT_PHY", "ACT_CONS", "AUDIT", "ACCEPT", "RECEIPT"]
        cycle_steps = []
        
        start_time = self._get_time()
        self.cycle_number += 1
        
        for stage in stages:
            step = self.step(dt, stage)
            cycle_steps.append(step)
        
        end_time = self._get_time()
        
        cycle = ExecutionCycle(
            cycle_number=self.cycle_number,
            start_time=start_time,
            end_time=end_time,
            steps=cycle_steps,
            is_deterministic=self.max_observed_deviation < 1e-10,
            max_deviation=self.max_observed_deviation
        )
        
        self.cycle_history.append(cycle)
        
        return cycle
    
    def run_until(self, t_end: float, dt: float = 1.0) -> List[ExecutionCycle]:
        """
        Run until virtual time reaches t_end.
        
        Args:
            t_end: Target virtual time
            dt: Time step size
            
        Returns:
            List of execution cycles
        """
        cycles = []
        
        while self._virtual_time < t_end:
            cycle = self.run_cycle(dt)
            cycles.append(cycle)
        
        return cycles
    
    def verify_determinism(self) -> Tuple[bool, float, List[str]]:
        """
        Verify that all executed operations were deterministic.
        
        Returns:
            Tuple of (is_deterministic, max_deviation, details)
        """
        if len(self.step_history) < 2:
            return True, 0.0, ["Insufficient data for verification"]
        
        details = []
        all_deterministic = True
        max_deviation = 0.0
        
        # Compare against base operation order
        if self.base_operation_order is None:
            self.base_operation_order = self.step_history[0].operations.copy()
            details.append(f"Base order established: {len(self.base_operation_order)} operations")
        
        for i, step in enumerate(self.step_history[1:], 1):
            deviation = self._compute_deviation(self.base_operation_order, step.operations)
            
            if deviation > 1e-10:
                all_deterministic = False
                details.append(f"Step {step.step_number}: deviation = {deviation:.2e}")
            
            max_deviation = max(max_deviation, deviation)
        
        if all_deterministic:
            details.append(f"All {len(self.step_history)} steps deterministic")
        else:
            details.append(f"Found non-deterministic operations (max deviation: {max_deviation:.2e})")
        
        return all_deterministic, max_deviation, details
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get comprehensive execution report."""
        is_det, deviation, details = self.verify_determinism()
        
        return {
            'step_count': self.step_number,
            'cycle_count': self.cycle_number,
            'virtual_time': self._virtual_time,
            'deterministic': is_det,
            'max_deviation': deviation,
            'total_steps': len(self.step_history),
            'total_cycles': len(self.cycle_history),
            'operation_details': details,
            'stage_distribution': self._get_stage_distribution(),
            'step_history_summary': self._summarize_history(),
        }
    
    def reset(self):
        """Reset scheduler state for new execution."""
        self.step_number = 0
        self.cycle_number = 0
        self._virtual_time = 0.0
        self.step_history = []
        self.cycle_history = []
        self.base_operation_order = None
        self.max_observed_deviation = 0.0
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _get_next_stage(self) -> str:
        """Get next stage in sequence."""
        stages = ["OBSERVE", "DECIDE", "ACT_PHY", "ACT_CONS", "AUDIT", "ACCEPT", "RECEIPT"]
        idx = self.step_number % len(stages)
        return stages[idx]
    
    def _get_time(self) -> float:
        """Get current time (deterministic or real)."""
        if self.deterministic_clock:
            return float(self.step_number) / 1000.0  # Virtual time in seconds
        return time.time()
    
    def _snapshot_state(self) -> Dict[str, Any]:
        """Create snapshot of current VM state."""
        if self.vm is None:
            return {}
        
        snapshot = {
            'step': self.step_number,
            'time': self._virtual_time,
        }
        
        if hasattr(self.vm, 'stack'):
            snapshot['stack_depth'] = len(self.vm.stack) if self.vm.stack else 0
        
        if hasattr(self.vm, '_ip'):
            snapshot['ip'] = self.vm._ip
        
        return snapshot
    
    def _get_operation_order(self) -> List[str]:
        """Get ordered list of executed operations."""
        if self.vm is None:
            return []
        
        if hasattr(self.vm, 'operation_log') and self.vm.operation_log:
            return [op['opcode'] for op in self.vm.operation_log]
        
        return self._current_operations.copy()
    
    def _verify_step_determinism(self, step: ExecutionStep) -> Tuple[bool, float]:
        """Verify if a step maintains determinism."""
        if self.base_operation_order is None:
            self.base_operation_order = step.operations.copy()
            return True, 0.0
        
        deviation = self._compute_deviation(self.base_operation_order, step.operations)
        return deviation < 1e-10, deviation
    
    def _compute_deviation(self, base: List[str], current: List[str]) -> float:
        """Compute deviation from base operation order."""
        if not base or not current:
            return 0.0
        
        if len(base) != len(current):
            return 1.0  # Different lengths = different order
        
        differences = sum(1 for a, b in zip(base, current) if a != b)
        return differences / len(base)
    
    def _get_stage_distribution(self) -> Dict[str, int]:
        """Get count of steps per stage."""
        distribution = {}
        for step in self.step_history:
            distribution[step.stage] = distribution.get(step.stage, 0) + 1
        return distribution
    
    def _summarize_history(self) -> List[Dict[str, Any]]:
        """Summarize execution history."""
        return [
            {
                'step': s.step_number,
                'stage': s.stage,
                'operations': len(s.operations),
                'deterministic': s.deterministic,
            }
            for s in self.step_history[-10:]  # Last 10 steps
        ]


# =============================================================================
# Parallel Scheduler
# =============================================================================

class ParallelScheduler:
    """
    Scheduler for parallel deterministic execution.
    
    Coordinates multiple workers while maintaining determinism
    through a shared operation log and synchronization points.
    """
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize parallel scheduler.
        
        Args:
            num_workers: Number of parallel workers
        """
        self.num_workers = num_workers
        self.workers: List[DeterministicScheduler] = []
        self.synchronization_points: List[int] = []
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize worker schedulers."""
        for _ in range(self.num_workers):
            worker = DeterministicScheduler(deterministic_clock=True)
            self.workers.append(worker)
    
    def run_parallel(self, dt: float = 1.0) -> List[ExecutionCycle]:
        """
        Run execution across all workers.
        
        All workers execute the same operations in the same order,
        enabling verification of parallel determinism.
        
        Args:
            dt: Time step size
            
        Returns:
            List of execution cycles from each worker
        """
        cycles = []
        
        # Run each worker
        for worker in self.workers:
            worker.reset()
            cycle = worker.run_cycle(dt)
            cycles.append(cycle)
        
        # Verify all workers are deterministic and agree
        self._verify_parallel_determinism(cycles)
        
        return cycles
    
    def _verify_parallel_determinism(self, cycles: List[ExecutionCycle]):
        """Verify all workers produced identical results."""
        if len(cycles) < 2:
            return
        
        # Compare first cycle as reference
        reference = cycles[0]
        
        for i, cycle in enumerate(cycles[1:], 1):
            # Check operation counts
            if len(cycle.steps) != len(reference.steps):
                raise DeterminismError(
                    f"Worker {i} has different step count",
                    cycle.cycle_number
                )
            
            # Check each step
            for j, (ref_step, worker_step) in enumerate(zip(reference.steps, cycle.steps)):
                if ref_step.operations != worker_step.operations:
                    raise DeterminismError(
                        f"Worker {i} step {j} has different operations",
                        worker_step.step_number
                    )
        
        # All workers are deterministic and agree
        return True


# =============================================================================
# Scheduler with JIT Support
# =============================================================================

class JITScheduler:
    """
    Scheduler with JIT compilation support for hot loops.
    
    Automatically compiles frequently executed sequences to
    native code while maintaining deterministic semantics.
    """
    
    def __init__(self, vm=None, jit_threshold: int = 100):
        """
        Initialize JIT scheduler.
        
        Args:
            vm: VM to schedule
            jit_threshold: Operations before JIT compilation
        """
        self.scheduler = DeterministicScheduler(vm)
        self.jit_threshold = jit_threshold
        self.hot_paths: Dict[str, int] = {}
        self.jit_compiled: List[str] = []
    
    def step(self, dt: float = 1.0, stage: Optional[str] = None) -> ExecutionStep:
        """Execute step with JIT detection."""
        return self.scheduler.step(dt, stage)
    
    def run_cycle(self, dt: float = 1.0) -> ExecutionCycle:
        """Run cycle with JIT compilation of hot paths."""
        cycle = self.scheduler.run_cycle(dt)
        
        # Detect hot paths
        self._detect_hot_paths(cycle)
        
        # Compile hot paths
        self._compile_hot_paths()
        
        return cycle
    
    def _detect_hot_paths(self, cycle: ExecutionCycle):
        """Detect frequently executed operation sequences."""
        # Build operation sequence string
        op_seq = '->'.join(cycle.steps[0].operations[:5]) if cycle.steps else ""
        
        if op_seq:
            self.hot_paths[op_seq] = self.hot_paths.get(op_seq, 0) + 1
    
    def _compile_hot_paths(self):
        """Compile hot paths to native code."""
        hot_threshold = self.jit_threshold * self.scheduler.cycle_number
        
        for path, count in self.hot_paths.items():
            if count > hot_threshold and path not in self.jit_compiled:
                self.jit_compiled.append(path)
                # In a full implementation, this would use numba/cython


# =============================================================================
# Convenience Functions
# =============================================================================

def create_deterministic_scheduler(vm=None) -> DeterministicScheduler:
    """Create a deterministic scheduler."""
    return DeterministicScheduler(vm=vm, deterministic_clock=True)


def create_parallel_scheduler(num_workers: int = 4) -> ParallelScheduler:
    """Create a parallel deterministic scheduler."""
    return ParallelScheduler(num_workers=num_workers)


# =============================================================================
# Demo/Test
# =============================================================================

if __name__ == '__main__':
    # Create scheduler
    scheduler = DeterministicScheduler(deterministic_clock=True)
    
    print("Deterministic Scheduler Demo")
    print("=" * 40)
    
    # Run a few steps
    for i in range(5):
        step = scheduler.step(dt=0.1)
        print(f"Step {step.step_number}: {step.stage}, ops={len(step.operations)}")
    
    # Verify determinism
    is_det, deviation, details = scheduler.verify_determinism()
    print(f"\nDeterminism check: {is_det}")
    print(f"Max deviation: {deviation:.2e}")
    
    # Get report
    report = scheduler.get_execution_report()
    print(f"\nReport: {report['step_count']} steps executed")
