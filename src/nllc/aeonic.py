import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from nllc.vm import VM
from nllc.nir import *
from cbtsv1.framework.receipt_schemas import OmegaReceipt

@dataclass
class ThreadBlock:
    name: str  # e.g., "PHY.step.act", "CONS.step.act", "PHY.step.audit"
    domain: str  # "PHY" or "CONS"
    phase: str  # "act" or "audit"
    block: BasicBlock  # The NIR basic block

    @property
    def vars_written(self) -> List[str]:
        """Extract variables written to by this block."""
        written = set()
        for inst in self.block.instructions:
            if isinstance(inst, StoreInst):
                written.add(inst.ptr.name)
            elif hasattr(inst, 'result'):
                written.add(inst.result.name)
        return list(written)

@dataclass
class AeonicReceipt:
    step_id: str
    dt: Optional[float]
    audit_pass: bool
    rollback_count: int
    module_id: str
    dep_closure_hash: str
    prev_receipt_id: Optional[str] = None
    receipt_id: str = ""

class AeonicScheduler:
    def __init__(self, vm: VM, max_retries: int = 3):
        self.vm = vm
        self.max_retries = max_retries
        self.receipts: List[AeonicReceipt] = []
        self.omega_receipts: List[OmegaReceipt] = []
        self.last_omega_id: Optional[str] = None

    def execute_steps(self, thread_blocks: List[ThreadBlock], steps: int, dt: Optional[float] = None) -> List[AeonicReceipt]:
        """Execute multiple steps, each with act then audit, with rollback on audit failure."""
        receipts = []
        for step in range(steps):
            step_receipts = self.execute_single_step(thread_blocks, step, dt)
            receipts.extend(step_receipts)
        return receipts

    def execute_single_step(self, thread_blocks: List[ThreadBlock], step_id: int, dt: Optional[float]) -> List[AeonicReceipt]:
        """Execute one step: act phases, then audit phases, rollback if audit fails."""
        if not self.vm.call_stack:
            raise ValueError("No active function context")

        env = self.vm.call_stack[-1]['env']

        # Separate act and audit threads
        act_threads = [tb for tb in thread_blocks if tb.phase == "act"]
        audit_threads = [tb for tb in thread_blocks if tb.phase == "audit"]

        # Snapshot state before act
        self.vm.snapshot_state()

        # Execute act threads in domain/scale/response order (assume order provided)
        for tb in act_threads:
            self.vm.execute_block(tb.block, env)

        # Execute audit threads, enforce no @phy writes
        audit_pass = True
        for tb in audit_threads:
            if tb.domain == "PHY":
                # Check if tries to write @phy variables - error if so
                phy_vars = [v for v in tb.vars_written if "@phy" in v]
                if phy_vars:
                    raise ValueError(f"Audit thread {tb.name} cannot write @phy variables: {phy_vars}")
            # Execute audit block
            result = self.vm.execute_block(tb.block, env)
            audit_pass = audit_pass and bool(result)  # Assume result is truthy for pass
            if not audit_pass:
                break

        rollback_count = 0
        if not audit_pass:
            # Rollback and retry up to max_retries
            for retry in range(self.max_retries):
                rollback_count += 1
                # Rollback to pre-act state
                self.vm.rollback_state()
                # Re-snapshot
                self.vm.snapshot_state()
                # Retry act
                for tb in act_threads:
                    self.vm.execute_block(tb.block, env)
                # Retry audit
                audit_pass_retry = True
                for tb in audit_threads:
                    if tb.domain == "PHY":
                        phy_vars = [v for v in tb.vars_written if "@phy" in v]
                        if phy_vars:
                            raise ValueError(f"Audit thread {tb.name} cannot write @phy variables: {phy_vars}")
                    result = self.vm.execute_block(tb.block, env)
                    audit_pass_retry = audit_pass_retry and bool(result)
                    if not audit_pass_retry:
                        break
                if audit_pass_retry:
                    audit_pass = True
                    break

        if audit_pass:
            # Pop the snapshot since succeeded
            self.vm.state_snapshots.pop()

        # Emit Omega receipt
        record = {
            "step_id": str(step_id),
            "dt": dt,
            "audit_pass": audit_pass,
            "rollback_count": rollback_count,
            "module_id": self.vm.module_id,
            "dep_closure_hash": self.vm.dep_closure_hash
        }
        omega_receipt = OmegaReceipt.create(prev=self.last_omega_id, tier="aeonic", record=record)
        self.omega_receipts.append(omega_receipt)
        self.last_omega_id = omega_receipt.id

        # Legacy AeonicReceipt for compatibility
        receipt = AeonicReceipt(
            step_id=str(step_id),
            dt=dt,
            audit_pass=audit_pass,
            rollback_count=rollback_count,
            module_id=self.vm.module_id,
            dep_closure_hash=self.vm.dep_closure_hash,
            prev_receipt_id=self.last_omega_id,
            receipt_id=omega_receipt.id
        )
        self.receipts.append(receipt)
        return [receipt]

    def get_receipts_json(self) -> str:
        return json.dumps([asdict(r) for r in self.receipts], sort_keys=True)