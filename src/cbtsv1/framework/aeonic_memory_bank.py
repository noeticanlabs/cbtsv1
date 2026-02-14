from dataclasses import dataclass
from typing import Any, List
import numpy as np
from src.aeonic.aeonic_clocks import AeonicClockPack
from src.core.aeonic_receipts import AeonicReceipts

@dataclass
class Record:
    key: str
    tier: int
    payload: Any
    bytes: int
    created_tau_s: int
    created_tau_l: int
    created_tau_m: int
    last_use_tau_s: int
    last_use_tau_l: int
    ttl_s: int
    ttl_l: int
    reuse_count: int
    recompute_cost_est: float
    risk_score: float
    tainted: bool
    regime_hashes: List[str]
    demoted: bool
    config_hash: str = None  # Optional config hash for scoped invalidation

class AeonicMemoryBank:
    def __init__(self, clock: AeonicClockPack, receipts: AeonicReceipts = None):
        self.clock = clock
        self.receipts = receipts
        self.tiers = {}  # tier: int -> dict of key: str -> Record
        self.total_bytes = 0
        self.tier_bytes = {}  # tier: int -> total bytes in tier

    def put(self, key: str, tier: int, payload: Any, bytes: int, ttl_s: int, ttl_l: int,
            recompute_cost_est: float, risk_score: float, tainted: bool, regime_hashes: List[str],
            demoted: bool = False, config_hash: str = None):
        """Insert or update a record in the specified tier."""
        if tier not in self.tiers:
            self.tiers[tier] = {}
            self.tier_bytes[tier] = 0

        record = Record(
            key=key,
            tier=tier,
            payload=payload,
            bytes=bytes,
            created_tau_s=self.clock.tau_s,
            created_tau_l=self.clock.tau_l,
            created_tau_m=self.clock.tau_m,
            last_use_tau_s=self.clock.tau_s,
            last_use_tau_l=self.clock.tau_l,
            ttl_s=ttl_s,
            ttl_l=ttl_l,
            reuse_count=0,
            recompute_cost_est=recompute_cost_est,
            risk_score=risk_score,
            tainted=tainted,
            regime_hashes=regime_hashes,
            demoted=demoted,
            config_hash=config_hash
        )

        if key in self.tiers[tier]:
            old_bytes = self.tiers[tier][key].bytes
            self.total_bytes -= old_bytes
            self.tier_bytes[tier] -= old_bytes

        self.tiers[tier][key] = record
        self.total_bytes += bytes
        self.tier_bytes[tier] += bytes

        if self.receipts:
            self.receipts.emit_event("MEM_PUT", {
                "key": key,
                "tier": tier,
                "bytes": bytes,
                "ttl_s": ttl_s,
                "ttl_l": ttl_l,
                "recompute_cost_est": recompute_cost_est,
                "risk_score": risk_score,
                "tainted": tainted,
                "regime_hashes": regime_hashes,
                "demoted": demoted,
                "tau_s": self.clock.tau_s,
                "tau_l": self.clock.tau_l,
                "tau_m": self.clock.tau_m
            })

    def get(self, key: str, tier: int = None):
        """Retrieve payload by key. If tier specified, only search that tier."""
        if tier is not None:
            if tier in self.tiers and key in self.tiers[tier]:
                record = self.tiers[tier][key]
                self._update_usage(record)
                if self.receipts:
                    self.receipts.emit_event("GET_HIT", {
                        "key": key,
                        "tier": tier,
                        "tau_s": self.clock.tau_s,
                        "tau_l": self.clock.tau_l
                    })
                return record.payload
        else:
            for t in sorted(self.tiers.keys()):  # search in order of tier?
                if key in self.tiers[t]:
                    record = self.tiers[t][key]
                    self._update_usage(record)
                    if self.receipts:
                        self.receipts.emit_event("GET_HIT", {
                            "key": key,
                            "tier": t,
                            "tau_s": self.clock.tau_s,
                            "tau_l": self.clock.tau_l
                        })
                    return record.payload
        if self.receipts:
            self.receipts.emit_event("GET_MISS", {
                "key": key,
                "tier": tier,
                "tau_s": self.clock.tau_s,
                "tau_l": self.clock.tau_l
            })
        return None

    def mark_tainted(self, key: str):
        """Mark the record with the given key as tainted in all tiers."""
        for tier in self.tiers:
            if key in self.tiers[tier]:
                self.tiers[tier][key].tainted = True
                if self.receipts:
                    self.receipts.emit_event("TAINT", {
                        "key": key,
                        "tier": tier,
                        "method": "mark_tainted",
                        "tau_s": self.clock.tau_s,
                        "tau_l": self.clock.tau_l
                    })

    def invalidate_by_regime(self, regime_hash: str, config_hash: str = None):
        """
        Mark all records containing the given regime_hash in their regime_hashes as tainted.
        
        Args:
            regime_hash: The regime hash to invalidate
            config_hash: Optional config hash for scoped invalidation (only invalidate records
                        matching this config_hash, or all if None)
        """
        for tier in self.tiers:
            for record in self.tiers[tier].values():
                if regime_hash in record.regime_hashes:
                    # Also match config_hash if provided
                    if config_hash is None or record.config_hash == config_hash:
                        record.tainted = True
                        if self.receipts:
                            self.receipts.emit_event("TAINT", {
                                "key": record.key,
                                "tier": tier,
                                "method": "invalidate_by_regime",
                                "regime_hash": regime_hash,
                                "config_hash": config_hash,
                                "tau_s": self.clock.tau_s,
                                "tau_l": self.clock.tau_l
                            })

    def _update_usage(self, record: Record):
        """Update last use times and reuse count."""
        record.last_use_tau_s = self.clock.tau_s
        record.last_use_tau_l = self.clock.tau_l
        record.reuse_count += 1

    def _compute_v_score(self, record: Record) -> float:
        """Compute valuation score for a record."""
        return record.reuse_count / (1 + record.risk_score + record.recompute_cost_est)

    def remove_record(self, tier: int, key: str):
        """Remove a record and update byte counts."""
        if tier in self.tiers and key in self.tiers[tier]:
            bytes = self.tiers[tier][key].bytes
            self.total_bytes -= bytes
            self.tier_bytes[tier] -= bytes
            del self.tiers[tier][key]
            if not self.tiers[tier]:
                del self.tiers[tier]
                del self.tier_bytes[tier]

    def maintenance_tick(self):
        """Perform maintenance: tick clock, expire TTL, demote Tier 2, evict by V score."""
        self.clock.tick_maintenance()

        # Expire records based on TTL
        to_remove = []
        for tier in self.tiers:
            for key, record in self.tiers[tier].items():
                if (self.clock.tau_s > record.created_tau_s + record.ttl_s or
                    self.clock.tau_l > record.created_tau_l + record.ttl_l):
                    to_remove.append((tier, key))
        for tier, key in to_remove:
            if self.receipts:
                self.receipts.emit_event("EXPIRE", {
                    "key": key,
                    "tier": tier,
                    "tau_s": self.clock.tau_s,
                    "tau_l": self.clock.tau_l
                })
            self.remove_record(tier, key)

        # Demote Tier 2 to summaries (move to Tier 3)
        if 2 in self.tiers:
            records = list(self.tiers[2].values())
            records.sort(key=lambda r: self._compute_v_score(r))
            to_demote = records[:5]  # demote up to 5 lowest V score
            for record in to_demote:
                record.demoted = True
                # Mixed precision for cold layers
                if isinstance(record.payload, np.ndarray) and record.payload.dtype in (np.float32, np.float64):
                    record.payload = record.payload.astype(np.float16)
                    record.bytes = record.payload.nbytes
                else:
                    record.payload = f"Demoted summary for key {record.key}"
                # Move to tier 3
                del self.tiers[2][record.key]
                self.tier_bytes[2] -= record.bytes
                self.total_bytes -= record.bytes
                record.tier = 3
                if 3 not in self.tiers:
                    self.tiers[3] = {}
                    self.tier_bytes[3] = 0
                self.tiers[3][record.key] = record
                self.total_bytes += record.bytes
                self.tier_bytes[3] += record.bytes
                if self.receipts:
                    self.receipts.emit_event("DEMOTE", {
                        "key": record.key,
                        "from_tier": 2,
                        "to_tier": 3,
                        "tau_s": self.clock.tau_s,
                        "tau_l": self.clock.tau_l
                    })

        # Evict by lowest V score (only if many records)
        total_records = sum(len(tier_dict) for tier_dict in self.tiers.values())
        if total_records > 10:  # Don't evict if few records
            all_records = []
            for tier in self.tiers:
                for record in self.tiers[tier].values():
                    all_records.append((tier, record))
            all_records.sort(key=lambda t_r: self._compute_v_score(t_r[1]))
            to_evict = all_records[:5]  # evict up to 5 lowest V score
            for tier, record in to_evict:
                if self.receipts:
                    self.receipts.emit_event("EVICT", {
                        "key": record.key,
                        "tier": tier,
                        "tau_s": self.clock.tau_s,
                        "tau_l": self.clock.tau_l
                    })
                self.remove_record(tier, record.key)