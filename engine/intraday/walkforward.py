"""
Walk-forward validation with purging and embargo.

The scanner's historical edge, as built so far, is measured on the SAME history
it is then applied to -- in-sample. In-sample hit-rates are optimistic: the
scenario parameters and the data overlap, so the number flatters itself. The
institutional fix is walk-forward out-of-sample (OOS) testing:

    1. Split the timeline into ordered folds.
    2. For each fold, measure the edge on PAST data (in-sample), then record how
       that edge actually performed on the NEXT fold (out-of-sample) -- data the
       measurement never touched.
    3. Aggregate OOS performance. That is the honest forward estimate.

Two leakage guards, both from Lopez de Prado's framework, adapted to intraday
session-level events:

    PURGE   -- drop in-sample events whose outcome window overlaps the OOS fold.
               Here each signal resolves within its own session, so the natural
               purge is at session granularity: no session straddles a fold
               boundary, and we additionally drop the last `purge_sessions`
               in-sample sessions before each OOS fold so an event near the
               boundary cannot leak.
    EMBARGO -- widen the in-sample guard band before each OOS fold by
               `embargo_sessions` (on top of purge). Outcomes resolve within
               their own session, so the only cross-session channel is prior-day
               levels (session N feeds N+1's prior-day high/low/close); a textbook
               post-test embargo has nothing to act on in an expanding window, so
               the sound equivalent is a wider pre-OOS isolation band. The IS
               window thus ends purge + embargo sessions before each OOS block.

The output reports in-sample vs out-of-sample net expectancy side by side. A
large IS->OOS decay is the signal that an edge is overfit. A scenario whose OOS
net expectancy is positive with a CI clear of zero is one you can actually
trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..gaps.statistics import bootstrap_mean_ci, estimate_proportion
from .backtest import SignalOutcome


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 5  # number of OOS folds
    purge_sessions: int = 1  # IS sessions dropped before each OOS fold
    embargo_sessions: int = 1  # extra IS guard-band sessions before each OOS (on top of purge)
    min_is_sessions: int = 20  # minimum in-sample sessions to measure on
    min_oos_events: int = 10  # minimum OOS events for a usable fold


@dataclass(frozen=True)
class FoldResult:
    fold: int
    is_sessions: int
    oos_sessions: int
    is_events: int
    oos_events: int
    is_net_expectancy_r: float
    oos_net_expectancy_r: float
    oos_net_ci: tuple[float, float]
    oos_win_rate: object  # ProportionEstimate


@dataclass(frozen=True)
class WalkForwardResult:
    scenario: str
    side: str
    folds: list[FoldResult]
    pooled_is_net_r: float
    pooled_oos_net_r: float
    pooled_oos_ci: tuple[float, float]
    oos_win_rate: object
    decay_r: float  # IS - OOS net expectancy (overfit gauge)
    n_oos_events: int
    trustworthy: bool  # OOS CI clear of zero AND enough events


def _session_order(outcomes: list[SignalOutcome]) -> list[pd.Timestamp]:
    """Unique session dates in chronological order."""
    sessions = sorted({pd.Timestamp(o.signal.session) for o in outcomes})
    return sessions


def walk_forward(
    outcomes: list[SignalOutcome],
    cfg: WalkForwardConfig | None = None,
) -> dict[tuple[str, str], WalkForwardResult]:
    """
    Run walk-forward per (scenario, side).

    `outcomes` are cost-aware SignalOutcomes (carry net_r and session). We bucket
    by session, build expanding-then-rolling IS/OOS folds with purge+embargo, and
    report IS vs OOS net expectancy.
    """
    cfg = cfg or WalkForwardConfig()
    if not outcomes:
        return {}

    # Group outcomes by (scenario, side).
    groups: dict[tuple[str, str], list[SignalOutcome]] = {}
    for o in outcomes:
        groups.setdefault((o.signal.scenario, o.signal.side.value), []).append(o)

    results: dict[tuple[str, str], WalkForwardResult] = {}
    for key, outs in groups.items():
        res = _walk_forward_one(outs, cfg)
        if res is not None:
            results[key] = res
    return results


def _net_r_array(outs: list[SignalOutcome]) -> np.ndarray:
    return np.array([o.net_r for o in outs], dtype=float)


def _walk_forward_one(
    outs: list[SignalOutcome], cfg: WalkForwardConfig
) -> WalkForwardResult | None:
    sessions = _session_order(outs)
    if len(sessions) < cfg.min_is_sessions + cfg.n_folds:
        return None

    by_session: dict[pd.Timestamp, list[SignalOutcome]] = {}
    for o in outs:
        by_session.setdefault(pd.Timestamp(o.signal.session), []).append(o)

    # Partition the tail of the timeline into n_folds contiguous OOS blocks,
    # leaving an initial in-sample base.
    n_sessions = len(sessions)
    base = max(
        cfg.min_is_sessions,
        n_sessions - cfg.n_folds * max(1, (n_sessions - cfg.min_is_sessions) // cfg.n_folds),
    )
    remaining = sessions[base:]
    if not remaining:
        return None
    fold_size = max(1, len(remaining) // cfg.n_folds)

    folds: list[FoldResult] = []
    pooled_is: list[float] = []
    pooled_oos: list[float] = []

    for f in range(cfg.n_folds):
        oos_start = base + f * fold_size
        oos_end = oos_start + fold_size if f < cfg.n_folds - 1 else n_sessions
        oos_sessions = sessions[oos_start:oos_end]
        if not oos_sessions:
            continue

        # In-sample = everything before oos_start, minus a guard band of
        # purge + embargo sessions (see module docstring: in an expanding,
        # within-session-resolving walk-forward the embargo widens the pre-OOS
        # isolation band rather than trailing the OOS block). Larger embargo =>
        # stricter isolation of the prior-day-level channel across the boundary.
        is_end = max(0, oos_start - cfg.purge_sessions - cfg.embargo_sessions)
        is_sessions = sessions[:is_end]
        if len(is_sessions) < cfg.min_is_sessions:
            continue

        is_outs = [o for s in is_sessions for o in by_session.get(s, [])]
        oos_outs = [o for s in oos_sessions for o in by_session.get(s, [])]
        if len(oos_outs) < cfg.min_oos_events:
            continue

        is_net = _net_r_array(is_outs)
        oos_net = _net_r_array(oos_outs)
        is_mean = float(np.mean(is_net)) if is_net.size else float("nan")
        oos_mean, oos_lo, oos_hi = bootstrap_mean_ci(oos_net)
        win = estimate_proportion(
            int((oos_net > 0).sum()),
            oos_net.size,
            prior_mean=0.5,
            prior_strength=10,
        )

        folds.append(
            FoldResult(
                fold=f,
                is_sessions=len(is_sessions),
                oos_sessions=len(oos_sessions),
                is_events=len(is_outs),
                oos_events=len(oos_outs),
                is_net_expectancy_r=is_mean,
                oos_net_expectancy_r=float(oos_mean),
                oos_net_ci=(float(oos_lo), float(oos_hi)),
                oos_win_rate=win,
            )
        )
        pooled_is.extend(is_net.tolist())
        pooled_oos.extend(oos_net.tolist())

    if not folds:
        return None

    pooled_oos_arr = np.array(pooled_oos, dtype=float)
    pooled_oos_mean, pooled_oos_lo, pooled_oos_hi = bootstrap_mean_ci(pooled_oos_arr)
    pooled_is_mean = float(np.mean(pooled_is)) if pooled_is else float("nan")
    pooled_win = estimate_proportion(
        int((pooled_oos_arr > 0).sum()),
        pooled_oos_arr.size,
        prior_mean=0.5,
        prior_strength=10,
    )
    trustworthy = pooled_oos_arr.size >= cfg.min_oos_events and pooled_oos_lo > 0.0

    return WalkForwardResult(
        scenario=outs[0].signal.scenario,
        side=outs[0].signal.side.value,
        folds=folds,
        pooled_is_net_r=pooled_is_mean,
        pooled_oos_net_r=float(pooled_oos_mean),
        pooled_oos_ci=(float(pooled_oos_lo), float(pooled_oos_hi)),
        oos_win_rate=pooled_win,
        decay_r=pooled_is_mean - float(pooled_oos_mean),
        n_oos_events=int(pooled_oos_arr.size),
        trustworthy=bool(trustworthy),
    )
