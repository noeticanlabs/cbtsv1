import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union
from receipt_schemas import MSolveReceipt, MStepReceipt, MOrchReceipt

@dataclass
class AeonicReceipts:
    log_file: str = "aeonic_receipts.jsonl"

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
        """Emit structured receipt with full schema."""
        receipt_dict = asdict(receipt)
        # Convert Kappa to dict for JSON serialization
        if 'kappa' in receipt_dict and hasattr(receipt.kappa, 'o'):
            receipt_dict['kappa'] = {'o': receipt.kappa.o, 's': receipt.kappa.s, 'mu': receipt.kappa.mu}

        event_type = type(receipt).__name__.upper()
        self.emit_event(event_type, receipt_dict)

    def snapshot_failure(self, attempts: list, filename: str = "failure_snapshot.jsonl"):
        """Persist failure snapshot for forensics."""
        with open(filename, 'w') as f:
            for attempt in attempts:
                attempt_dict = asdict(attempt)
                if 'kappa' in attempt_dict and hasattr(attempt.kappa, 'o'):
                    attempt_dict['kappa'] = {'o': attempt.kappa.o, 's': attempt.kappa.s, 'mu': attempt.kappa.mu}
                f.write(json.dumps(attempt_dict, default=str) + '\n')