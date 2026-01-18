import dataclasses
from typing import List, Optional
import hashlib
import json
from src.module.manifest import compute_module_id

@dataclasses.dataclass
class NamespaceReceipt:
    module_id: str
    dep_closure_hash: str
    policy_hash: str
    compiler: str
    target: str
    timestamp: str  # ISO 8601 UTC

@dataclasses.dataclass
class RunReceipt:
    step_id: str
    trace_digest: str
    prev: Optional[str]
    id: str

def create_namespace_receipt(manifest: dict, dep_closure_hash: str, policy_hash: str, compiler: str, target: str, timestamp: str) -> NamespaceReceipt:
    module_id = compute_module_id(manifest, dep_closure_hash, policy_hash)
    return NamespaceReceipt(module_id=module_id, dep_closure_hash=dep_closure_hash, policy_hash=policy_hash, compiler=compiler, target=target, timestamp=timestamp)

def create_run_receipt(step_id: str, trace_digest: str, prev: Optional[str]) -> RunReceipt:
    data = f"{step_id}|{trace_digest}|{prev or ''}"
    receipt_id = hashlib.sha256(data.encode('utf-8')).hexdigest()
    return RunReceipt(step_id=step_id, trace_digest=trace_digest, prev=prev, id=receipt_id)

def verify_receipt_chain(receipts: List[RunReceipt]) -> bool:
    for i, receipt in enumerate(receipts):
        # Recompute id
        data = f"{receipt.step_id}|{receipt.trace_digest}|{receipt.prev or ''}"
        computed_id = hashlib.sha256(data.encode('utf-8')).hexdigest()
        if computed_id != receipt.id:
            return False
        # Check prev link
        if i > 0:
            if receipt.prev != receipts[i-1].id:
                return False
    return True