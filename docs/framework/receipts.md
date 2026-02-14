# Receipts

## Overview

Receipts are immutable records of solver operations. They provide:
- Audit trail for computations
- Hashchain integrity
- Verification of solver state at each step

## Receipt Structure

```python
@dataclass
class Receipt:
    timestamp: str          # ISO 8601 timestamp
    step: int               # Timestep number
    operation: str          # Operation type
    inputs: Dict            # Input hashes
    outputs: Dict           # Output hashes
    hashchain: str          # Chain hash
    metadata: Dict           # Additional metadata
```

## Types of Receipts

1. **Step Receipt**: Each timestep computation
2. **Constraint Receipt**: Constraint violation data
3. **Gauge Receipt**: Gauge transformation records
4. **Ledger Receipt**: State ledger updates

## Usage

```python
from cbtsv1.framework.receipts import AeonicReceipts

receipts = AeonicReceipts()
receipts.emit(step=100, operation="timestep", data=state)
```

## Hashchain

Each receipt includes a hashchain entry linking to the previous receipt, ensuring integrity.
