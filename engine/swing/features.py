"""
Swing event extraction over daily bars.

Scanner #2. The structural substrate (multi-scale zigzag legs) is REUSED from the
intraday layer unchanged -- a daily zigzag finds swing legs exactly as a 1-min
zigzag finds intraday legs. Only the FEATURES differ: a swing has no VWAP or
opening range, so context is moving-average / range / volatility based.

Causality is identical to the intraday fix: a swing event's decision bar is the
pivot's CONFIRMATION bar (leg.decision_index), never the extreme (which is known
only in hindsight), and every feature is computed from bars up to that bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.structural_unit import StructuralUnit
from ..ml.features import EventFeatures, _causal_atr
from ..session.pivots import legs_from_pivots, merge_insignificant_swings, pivots_at_scale

DEFAULT_SWING_SCALE = 1.5  # zigzag threshold in ATR units; a "swing" on daily bars


def _swing_legs(unit: StructuralUnit, scale_atr: float):
    """Clean swing legs at one scale: merge insignificant counter-swings, drop
    zero-duration legs. Carries each leg's pivot confirmation bar (causal)."""
    pivots = pivots_at_scale(unit, scale_atr)
    legs = merge_insignificant_swings(legs_from_pivots(unit, pivots))
    return [lg for lg in legs if lg.end_index > lg.start_index]


def _swing_features(unit: StructuralUnit, leg, idx: int, atr: float) -> dict[str, float]:
    """Causal swing context at bar `idx` (the leg's confirmation bar).

    MA/range/volatility based -- no VWAP. "At a new high/low so far" uses RUNNING
    extremes (causal), never the full-window extreme. All windows are trailing
    (`.tail`) and therefore use only bars <= idx.
    """
    b = unit.bars
    upto = b.iloc[: idx + 1]
    close = upto["close"]
    c = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma50 = float(close.tail(50).mean())
    run_high = float(upto["high"].max())
    run_low = float(upto["low"].min())
    look = min(60, idx + 1)
    win_high = float(upto["high"].tail(look).max())
    win_low = float(upto["low"].tail(look).min())
    rng = win_high - win_low
    rwin = close.tail(min(20, len(close))).to_numpy(dtype=float)
    rets = np.diff(np.log(np.clip(rwin, 1e-9, None)))
    realized = float(np.std(rets)) if rets.size else 0.0
    tol = 0.05 * atr
    return {
        "dist_from_ma20_atr": (c - ma20) / atr,
        "dist_from_ma50_atr": (c - ma50) / atr,
        "ma20_vs_ma50_atr": (ma20 - ma50) / atr,
        "range_pos_60": (c - win_low) / rng if rng > 0 else 0.5,
        "dist_from_runhigh_atr": (c - run_high) / atr,
        "dist_from_runlow_atr": (c - run_low) / atr,
        "realized_vol": realized,
        "leg_magnitude_atr": leg.magnitude / atr if atr > 0 else 0.0,
        "leg_bars": float(leg.bars),
        "leg_is_up": 1.0 if leg.direction == "up" else 0.0,
        "leg_above_ma50": 1.0 if leg.end_price > ma50 else 0.0,
        "at_running_high": 1.0 if leg.end_price >= run_high - tol else 0.0,
        "at_running_low": 1.0 if leg.end_price <= run_low + tol else 0.0,
        "confirmation_lag_bars": float(idx - leg.end_index),
    }


def extract_swing_events(
    unit: StructuralUnit, scale_atr: float = DEFAULT_SWING_SCALE
) -> list[EventFeatures]:
    """Every completed swing leg as a causal feature vector (decision bar =
    confirmation). Empty if the window has no structure at this scale."""
    n = len(unit)
    b = unit.bars
    closes = b["close"].to_numpy(dtype=float)
    times = b["datetime"].to_numpy()
    out: list[EventFeatures] = []
    for leg in _swing_legs(unit, scale_atr):
        idx = min(leg.decision_index, n - 1)
        atr = _causal_atr(unit, idx)
        event_time = (
            leg.confirmed_time if leg.confirmed_time is not None else pd.Timestamp(times[idx])
        )
        out.append(
            EventFeatures(
                symbol=unit.symbol,
                # the EVENT's own bar date (not the window's as-of date) -- needed
                # for correct time-ordering and the forward day-count.
                date=pd.Timestamp(event_time).normalize(),
                event_type="swing_leg",
                event_index=idx,
                event_time=event_time,
                event_price=float(closes[idx]),
                features=_swing_features(unit, leg, idx, atr),
            )
        )
    out.sort(key=lambda e: e.event_index)
    return out
