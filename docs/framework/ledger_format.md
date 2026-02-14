# Ledger Format

## Overview

The ledger stores the solver state at each timestep, providing:
- Immutable state snapshots
- Hashchain verification
- Reproducibility

## Ledger Entry

```python
@dataclass
class LedgerEntry:
    step: int
    timestamp: str
    state_hash: str
    previous_hash: str
    data: Dict[str, np.ndarray]
    metadata: Dict
```

## State Fields

| Field | Type | Description |
|-------|------|-------------|
| gamma | (3,3) tensor | 3-metric |
| K | (3,3) tensor | Extrinsic curvature |
| alpha | scalar | Lapse |
| beta | (3,) vector | Shift |
| dt | float | Timestep |

## Hashchain

Each ledger entry links to the previous via hashchain:
```
entry[i].previous_hash = hash(entry[i-1])
```

## Storage

Ledgers are stored as:
- JSON for human readability
- Binary for efficiency (future)
