"""
Signal journal: live forward testing (log now, resolve later).

batch_rank fires signals today; this records them append-only (JSONL) and, once
enough bars have printed, resolves each against its OWN later bars into a realized
R-multiple — conservative stop-first, the same rule as labeling. The summary then
reports realized live performance next to the validated edge each signal carried.

No lookahead by construction: a signal is resolved only with bars STRICTLY AFTER
its decision time. The rigorous validated-vs-realized DECAY on history is the
forward-test runner's job; the journal is the live-accumulation mechanism.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ml.signals import Signal


def resolve_entry(entry: dict, bars: pd.DataFrame) -> dict:
    """Resolve one open entry against later `bars` (OHLC + datetime).

    Returns the entry updated with realized_r / exit_reason / bars_held /
    resolved_time / status='resolved' when the bracket resolves; unchanged (still
    'open') if not enough bars have printed yet.
    """
    if entry.get("status") == "resolved":
        return entry
    b = bars.sort_values("datetime").reset_index(drop=True)
    dt = pd.Timestamp(entry["timestamp"])
    after = b[pd.to_datetime(b["datetime"]) > dt].reset_index(drop=True)
    if after.empty:
        return entry

    long = entry["direction"] == "long"
    e = float(entry["entry"])
    stop = float(entry["stop"])
    target = float(entry["target"])
    risk = abs(e - stop)
    rr = float(entry.get("rr") or (abs(target - e) / risk if risk > 0 else 0.0))
    max_bars = int(entry.get("max_bars") or 0)

    hi = after["high"].to_numpy(dtype=float)
    lo = after["low"].to_numpy(dtype=float)
    cl = after["close"].to_numpy(dtype=float)
    n = len(after)
    scan = min(max_bars, n) if max_bars > 0 else n
    times = after["datetime"].astype(str).to_numpy()

    def _resolved(realized_r, reason, j):
        return {
            **entry,
            "status": "resolved",
            "realized_r": float(realized_r),
            "exit_reason": reason,
            "bars_held": int(j + 1),
            "resolved_time": str(times[j]),
        }

    for j in range(scan):
        hit_t = hi[j] >= target if long else lo[j] <= target
        hit_s = lo[j] <= stop if long else hi[j] >= stop
        if hit_s:  # both-in-one-bar resolves here too -> stop first (conservative)
            return _resolved(-1.0, "stop", j)
        if hit_t:
            return _resolved(rr, "target", j)

    # No hit within the scanned bars.
    if max_bars > 0 and n >= max_bars:  # full horizon elapsed -> mark to market
        exit_px = cl[max_bars - 1]
        move = (exit_px - e) if long else (e - exit_px)
        return _resolved(move / risk if risk > 0 else 0.0, "horizon", max_bars - 1)
    return entry  # horizon not yet elapsed -> still open


@dataclass(frozen=True)
class ForwardSummary:
    n_open: int
    n_resolved: int
    realized_mean_r: float  # net of cost_r
    realized_hit_rate: float
    mean_validated_edge_r: float  # avg oos_edge_r carried by the resolved signals
    by_reason: dict[str, int]


@dataclass
class SignalJournal:
    """Append-only JSONL store of fired signals + their resolution."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def log(self, signals: Iterable[Signal], scanner: str = "intraday") -> int:
        n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for s in signals:
                rec = s.as_dict()  # carries entry/stop/target/rr/max_bars/oos_edge_r
                rec["scanner"] = scanner
                rec["status"] = "open"
                f.write(json.dumps(rec) + "\n")
                n += 1
        return n

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def resolve_all(self, bars_by_symbol: Mapping[str, pd.DataFrame]) -> list[dict]:
        """Resolve every open entry whose symbol has bars supplied; persists."""
        rows = self.entries()
        out = []
        for e in rows:
            if e.get("status") == "open" and e.get("symbol") in bars_by_symbol:
                e = resolve_entry(e, bars_by_symbol[e["symbol"]])
            out.append(e)
        with self.path.open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        return out

    def summary(self, cost_r: float = 0.05) -> ForwardSummary:
        rows = self.entries()
        resolved = [r for r in rows if r.get("status") == "resolved"]
        n_open = len(rows) - len(resolved)
        # Carried validated edge across ALL logged signals (informative even before
        # anything resolves); realized is over resolved trades only.
        carried = np.array([float(r.get("oos_edge_r", 0.0)) for r in rows], dtype=float)
        mean_validated = float(carried.mean()) if carried.size else 0.0
        if not resolved:
            return ForwardSummary(n_open, 0, 0.0, 0.0, mean_validated, {})
        net = np.array([float(r["realized_r"]) - cost_r for r in resolved], dtype=float)
        by_reason: dict[str, int] = {}
        for r in resolved:
            by_reason[r.get("exit_reason", "?")] = by_reason.get(r.get("exit_reason", "?"), 0) + 1
        return ForwardSummary(
            n_open=n_open,
            n_resolved=len(resolved),
            realized_mean_r=float(net.mean()),
            realized_hit_rate=float((net > 0).mean()),
            mean_validated_edge_r=mean_validated,
            by_reason=by_reason,
        )
