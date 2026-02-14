# Validation Tests

## Test Suites

### Minkowski Stability Test
- **Purpose**: Verify constraints remain zero for flat spacetime
- **Expected**: H ≈ 0, Mⁱ ≈ 0 to machine precision
- **Config**: `config/gr_rhs.nsc`

### Schwarzschild Slice Test
- **Purpose**: Verify constraint violation remains small away from singularity
- **Expected**: Constraints ~10^-6 at modest resolution
- **Initial Data**: Isotropic Schwarzschild

### Convergence Tests
- **Purpose**: Verify numerical convergence order
- **Method**: MMS (Method of Manufactured Solutions)
- **Expected**: N-th order convergence for N-th order methods

### Gauge Invariance Test
- **Purpose**: Verify solution independent of gauge choice
- **Method**: Compare runs with different gauge conditions

## Running Tests

```bash
# Run Minkowski test
python scripts/run_nllc_gr_test.py --test minkowski

# Run convergence test
python scripts/mms_parameter_sweep.py

# Run full suite
pytest tests/test_*mms*.py tests/test_*constraint*.py
```

## Test Files

- [`tests/test_mms_*.py`](../../tests/test_mms_*.py) - MMS convergence tests
- [`tests/test_*constraint*.py`](../../tests/test_*constraint*.py) - Constraint monitoring
- [`tests/test_*gauge*.py`](../../tests/test_*gauge*.py) - Gauge tests
