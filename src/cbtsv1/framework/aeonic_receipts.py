import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union, Optional
from src.receipts.receipt_schemas import MSolveReceipt, MStepReceipt, MOrchReceipt

def canonical_json_dumps(obj):
    """Triaxis v1.2 canonical JSON serialization: sorted keys, no whitespace, floats as decimal strings."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)

@dataclass
class AeonicReceipts:
    log_file: str = "aeonic_receipts.jsonl"
    last_id: Optional[str] = None

    def emit_event(self, event_type: str, details: Dict[str, Any]):
        """Emit a JSONL log entry for a memory event."""
        entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            **details
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def emit_structured_receipt(self, receipt: Union[MSolveReceipt, MStepReceipt, MOrchReceipt]):
        """Emit unified OmegaReceipt with hash chaining."""
        from src.receipts.receipt_schemas import OmegaReceipt

        # Determine tier based on type
        tier_map = {
            MSolveReceipt: "msolve",
            MStepReceipt: "mstep",
            MOrchReceipt: "morch"
        }
        tier = tier_map[type(receipt)]

        # Convert receipt to dict for record
        receipt_dict = asdict(receipt)
        if 'kappa' in receipt_dict and hasattr(receipt.kappa, 'o'):
            receipt_dict['kappa'] = {'o': receipt.kappa.o, 's': receipt.kappa.s, 'mu': receipt.kappa.mu}

        # Create unified receipt
        omega_receipt = OmegaReceipt.create(prev=self.last_id, tier=tier, record=receipt_dict)

        # Emit
        self.emit_event("OMEGARECEIPT", asdict(omega_receipt))

        # Update last_id
        self.last_id = omega_receipt.id

    def snapshot_failure(self, attempts: list, filename: str = "failure_snapshot.jsonl"):
        """Persist failure snapshot for forensics."""
        with open(filename, 'w') as f:
            for attempt in attempts:
                attempt_dict = asdict(attempt)
                if 'kappa' in attempt_dict and hasattr(attempt.kappa, 'o'):
                    attempt_dict['kappa'] = {'o': attempt.kappa.o, 's': attempt.kappa.s, 'mu': attempt.kappa.mu}
                f.write(json.dumps(attempt_dict, default=str) + '\n')