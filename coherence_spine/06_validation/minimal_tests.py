"""
Minimal Coherence Test Harness (stdlib only)

Validates:
  - deterministic receipt hashing + hash chaining
  - a simple gate+rail loop on x' = -lambda x (explicit Euler + dt deflation)
Run:
  python minimal_tests.py
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass
class GatePolicy:
    phys_fail: float = 1e-2
    phys_warn: float = 7.5e-3
    dt_min: float = 1e-4
    dt_alpha: float = 0.8
    retry_cap: int = 6

    def evaluate(self, phys_rms: float, nan_free: bool = True) -> dict:
        hard = {"nan_free": "pass" if nan_free else "fail"}
        soft = {"phys_rms": "pass"}
        decision = "accept"

        if not nan_free:
            decision = "abort"
        else:
            if phys_rms > self.phys_fail:
                soft["phys_rms"] = "fail"
                decision = "reject"
            elif phys_rms > self.phys_warn:
                soft["phys_rms"] = "warn"
                decision = "accept"

        return {"hard": hard, "soft": soft, "decision": decision}


def step_explicit_euler(x: float, lam: float, dt: float) -> float:
    return x + dt * (-lam * x)


def defect_residual(x: float, x_next: float, lam: float, dt: float) -> float:
    return (x_next - x) / dt + lam * x


def run():
    lam = 2.0
    x = 1.0
    t = 0.0
    dt = 0.05
    gp = GatePolicy()

    prev_hash = "sha256:" + "0" * 64
    receipts = []
    accepted_steps = 0

    for step_id in range(40):
        retries = 0
        while True:
            x_prop = step_explicit_euler(x, lam, dt)
            defect = defect_residual(x, x_prop, lam, dt)
            phys_rms = abs(defect)
            nan_free = math.isfinite(x_prop) and math.isfinite(phys_rms)

            verdicts = gp.evaluate(phys_rms, nan_free=nan_free)

            rails = []
            decision = verdicts["decision"]

            if verdicts["hard"]["nan_free"] == "fail":
                decision = "abort"

            if decision == "reject":
                retries += 1
                if retries > gp.retry_cap:
                    rails.append({"name": "rollback", "reason": "retry_cap"})
                    x_prop = x
                else:
                    dt_new = max(gp.dt_min, gp.dt_alpha * dt)
                    rails.append(
                        {
                            "name": "dt_deflate",
                            "dt_before": dt,
                            "dt_after": dt_new,
                            "reason": "phys_rms",
                        }
                    )
                    dt = dt_new

            receipt = {
                "header": {
                    "run_id": "RUN-MINIMAL-ODE",
                    "step_id": step_id,
                    "t": t,
                    "dt": dt,
                    "timezone": "America/Chicago",
                },
                "state": {
                    "hash_before": "sha256:" + sha256_hex(b"before" + canonical_json({"x": x})),
                    "hash_after": "sha256:" + sha256_hex(b"after" + canonical_json({"x": x_prop})),
                    "summary": {"x": x_prop},
                },
                "metrics": {
                    "residuals": {"phys_rms": phys_rms},
                    "debt": {"C_total": phys_rms**2, "C_blocks": {"phys": phys_rms**2}},
                    "invariants": {"nan_free": nan_free},
                },
                "gates": {"hard": verdicts["hard"], "soft": verdicts["soft"], "decision": decision},
                "actions": {"rails": rails, "retries_used": retries, "notes": []},
                "provenance": {
                    "lexicon_terms_used": ["LoC_axiom", "CTL_clock"],
                    "layers": ["L0", "L2", "L4"],
                    "code_version": "v1.0.0",
                    "seed": 0,
                },
                "integrity": {"prev_hash": prev_hash, "this_hash": ""},
            }

            receipt_for_hash = dict(receipt)
            receipt_for_hash["integrity"] = dict(receipt["integrity"])
            receipt_for_hash["integrity"]["this_hash"] = ""
            this_hash = "sha256:" + sha256_hex(canonical_json(receipt_for_hash) + prev_hash.encode("utf-8"))
            receipt["integrity"]["this_hash"] = this_hash
            prev_hash = this_hash
            receipts.append(receipt)

            if decision == "accept":
                x = x_prop
                t += dt
                accepted_steps += 1
                break
            if decision == "abort":
                break
            if decision == "reject" and retries > gp.retry_cap:
                break

    prev = "sha256:" + "0" * 64
    for receipt in receipts:
        receipt_for_hash = dict(receipt)
        receipt_for_hash["integrity"] = dict(receipt["integrity"])
        receipt_for_hash["integrity"]["this_hash"] = ""
        chk = "sha256:" + sha256_hex(canonical_json(receipt_for_hash) + prev.encode("utf-8"))
        assert chk == receipt["integrity"]["this_hash"], "Hash chain mismatch"
        prev = receipt["integrity"]["this_hash"]

    worst = max(receipts, key=lambda rr: rr["metrics"]["residuals"]["phys_rms"])
    report = {
        "accepted_steps": accepted_steps,
        "total_receipts": len(receipts),
        "max_phys_rms": worst["metrics"]["residuals"]["phys_rms"],
        "max_debt": worst["metrics"]["debt"]["C_total"],
        "final_t": t,
        "final_dt": dt,
        "final_x": x,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
