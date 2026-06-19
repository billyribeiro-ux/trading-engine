"""
Long-term position events over weekly bars (Scanner #3).

Same causal discipline and reused zigzag as intraday/swing; the new dimension is
RELATIVE STRENGTH vs a benchmark — the defining feature of a portfolio screen
(does this name lead or lag the market?). RS is computed causally over a trailing
window and is exactly the kind of signal that matters for inverse/leveraged ETFs:
SQQQ scanned on its OWN bars, with RS vs QQQ as a feature, never as a substitute
for its label.

Fundamentals (earnings/margins/sector rotation) are the deliberate NEXT layer:
they carry their own causality trap (known only as of the report/filing date, not
the fiscal period) and are added as as-of-joined features later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.structural_unit import StructuralUnit
from ..ml.features import EventFeatures, _causal_atr
from ..session.pivots import legs_from_pivots, merge_insignificant_swings, pivots_at_scale

DEFAULT_POSITION_SCALE = 2.0  # coarse zigzag: multi-week/-month legs
RS_WEEKS = 13  # trailing relative-strength window (~one quarter)


def _position_legs(unit: StructuralUnit, scale_atr: float):
    pivots = pivots_at_scale(unit, scale_atr)
    legs = merge_insignificant_swings(legs_from_pivots(unit, pivots))
    return [lg for lg in legs if lg.end_index > lg.start_index]


def _position_features(
    unit: StructuralUnit, leg, idx: int, atr: float, bench_close: np.ndarray | None
) -> dict[str, float]:
    b = unit.bars
    upto = b.iloc[: idx + 1]
    close = upto["close"]
    c = float(close.iloc[-1])
    ma10 = float(close.tail(10).mean())  # ~50-day
    ma30 = float(close.tail(30).mean())  # ~150-day
    run_high = float(upto["high"].max())
    run_low = float(upto["low"].min())
    look = min(52, idx + 1)
    win_high = float(upto["high"].tail(look).max())
    win_low = float(upto["low"].tail(look).min())
    rng = win_high - win_low
    tol = 0.05 * atr
    feats = {
        "dist_from_ma10_atr": (c - ma10) / atr,
        "dist_from_ma30_atr": (c - ma30) / atr,
        "ma10_vs_ma30_atr": (ma10 - ma30) / atr,
        "range_pos_52w": (c - win_low) / rng if rng > 0 else 0.5,
        "dist_from_runhigh_atr": (c - run_high) / atr,
        "dist_from_runlow_atr": (c - run_low) / atr,
        "leg_magnitude_atr": leg.magnitude / atr if atr > 0 else 0.0,
        "leg_bars": float(leg.bars),
        "leg_is_up": 1.0 if leg.direction == "up" else 0.0,
        "leg_above_ma30": 1.0 if leg.end_price > ma30 else 0.0,
        "at_running_high": 1.0 if leg.end_price >= run_high - tol else 0.0,
        "at_running_low": 1.0 if leg.end_price <= run_low + tol else 0.0,
        "confirmation_lag_bars": float(idx - leg.end_index),
    }
    # Relative strength vs the benchmark over RS_WEEKS (causal: bars idx-N..idx).
    n = RS_WEEKS
    if bench_close is not None and idx - n >= 0:
        past = float(close.iloc[idx - n])
        bpast, bnow = bench_close[idx - n], bench_close[idx]
        if past > 0 and np.isfinite(bpast) and bpast > 0 and np.isfinite(bnow):
            sym_ret = c / past - 1.0
            bench_ret = bnow / bpast - 1.0
            feats["rs_vs_bench_13w"] = sym_ret - bench_ret
            feats["bench_trend_13w"] = bench_ret
    return feats


def extract_position_events(
    unit: StructuralUnit,
    scale_atr: float = DEFAULT_POSITION_SCALE,
    bench_close: np.ndarray | None = None,
    fundamentals=None,
) -> list[EventFeatures]:
    """Every completed long-term leg as a causal feature vector (decision bar =
    confirmation). RS added when a benchmark is supplied; fundamental features
    added AS OF the decision bar's date when a FundamentalSeries is supplied (a
    report filed after the bar can never be used)."""
    n = len(unit)
    b = unit.bars
    closes = b["close"].to_numpy(dtype=float)
    times = b["datetime"].to_numpy()
    out: list[EventFeatures] = []
    for leg in _position_legs(unit, scale_atr):
        idx = min(leg.decision_index, n - 1)
        atr = _causal_atr(unit, idx)
        event_time = (
            leg.confirmed_time if leg.confirmed_time is not None else pd.Timestamp(times[idx])
        )
        feats = _position_features(unit, leg, idx, atr, bench_close)
        if fundamentals is not None:
            feats.update(fundamentals.asof(event_time))  # as-of join — causal
        out.append(
            EventFeatures(
                symbol=unit.symbol,
                date=unit.date,
                event_type="position_leg",
                event_index=idx,
                event_time=event_time,
                event_price=float(closes[idx]),
                features=feats,
            )
        )
    out.sort(key=lambda e: e.event_index)
    return out
