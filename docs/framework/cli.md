# CLI Usage

## Running the Solver

```bash
# Run with default config
python -m cbtsv1.solvers.gr.gr_solver

# Run with custom config
python -m cbtsv1.solvers.gr.gr_solver --config config/gr_rhs.nsc

# Run specific test
python scripts/run_nllc_gr_test.py --test minkowski
```

## Options

| Option | Description |
|--------|-------------|
| `--config` | Path to NSC config file |
| `--output` | Output directory for ledgers |
| `--steps` | Number of timesteps |
| `--resolution` | Grid resolution |

## MMS Testing

```bash
# Run convergence test
python scripts/mms_parameter_sweep.py --order 4

# Run fast sweep
python scripts/mms_fast_sweep.py
```

## Inspecting Results

```bash
# View receipts
python scripts/inspect_receipts.py artifacts/aeonic_receipts.jsonl
```
