"""
The gauntlet: the full evidence recipe as one reusable verdict.

Bundles the two independent forward tests a candidate must pass:
  1. bake_off  — single out-of-time block, FDR across models, day-guarded.
  2. rolling   — persistence across N sequential windows (regime defense).
A model PASSES only if it is BOTH promoted in the bake-off AND robust in the
rolling test. Run this on any (universe, scanner-config, direction) — including a
fresh symbol set — to decide whether an edge is real. "Not passed" is the common,
honest result. This is the recipe behind the GBT-swing-long finding; reuse it
verbatim for every future candidate so the bar never silently drops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..ml.dataset import feature_columns
from ..ml.signals import ScannerConfig
from .bakeoff import ModelVersion, VersionResult, bake_off, render_bakeoff
from .pooled import build_pooled_frame
from .runner import RollingForwardResult, rolling_forward_test


@dataclass(frozen=True)
class GauntletVerdict:
    direction: str
    n_events: int
    distinct_days: int
    bakeoff: tuple[VersionResult, ...]
    rolling: dict[str, RollingForwardResult]
    passed: bool  # some model both PROMOTED (bake-off) and ROBUST (rolling)


def run_gauntlet(
    symbols: Sequence[str],
    config: ScannerConfig,
    *,
    models: Sequence[str] = ("logistic", "gbt"),
    direction: str = "long",
    cutoff_frac: float = 0.7,
    fdr: float = 0.10,
    min_holdout_signals: int = 20,
    min_holdout_days: int = 15,
    n_windows: int = 5,
    seed: int = 0,
) -> GauntletVerdict:
    pooled = build_pooled_frame(symbols, config, direction=direction)
    if pooled.empty:
        return GauntletVerdict(direction, 0, 0, (), {}, False)
    fc = feature_columns(pooled)
    hb = config.bracket.max_bars

    rows = bake_off(
        pooled,
        fc,
        [ModelVersion(m, m) for m in models],
        horizon_bars=hb,
        cutoff_frac=cutoff_frac,
        cost_r=config.cost_r,
        fdr=fdr,
        min_holdout_signals=min_holdout_signals,
        min_holdout_days=min_holdout_days,
        seed=seed,
    )
    rolling = {
        m: rolling_forward_test(
            pooled,
            fc,
            n_windows=n_windows,
            model_kind=m,
            horizon_bars=hb,
            cost_r=config.cost_r,
            min_holdout_signals=min_holdout_signals,
            min_holdout_days=max(8, min_holdout_days // 2),
            seed=seed,
        )
        for m in models
    }
    promoted = {r.version for r in rows if r.promoted}
    passed = any(m in promoted and rolling[m].robust for m in models)
    return GauntletVerdict(
        direction=direction,
        n_events=len(pooled),
        distinct_days=int(pd.to_datetime(pooled["date"]).dt.normalize().nunique()),
        bakeoff=tuple(rows),
        rolling=rolling,
        passed=passed,
    )


def render_gauntlet(v: GauntletVerdict) -> str:
    lines = [
        f"GAUNTLET [{v.direction}]  events={v.n_events}  distinct_days={v.distinct_days}  "
        f"PASSED={v.passed}",
        render_bakeoff(v.bakeoff),
    ]
    for m, rr in v.rolling.items():
        lines.append(
            f"  rolling[{m}]: persisted={rr.n_persisted}/{rr.n_windows} "
            f"pooled_R={rr.pooled_realized_edge_r:+.3f} p={rr.pooled_realized_p:.3f} "
            f"days={rr.pooled_holdout_days} robust={rr.robust}"
        )
    if not v.passed:
        lines.append("  -> no model passes the full gauntlet. Honest null; do not trade.")
    return "\n".join(lines)
