"""
Event feature extraction for the self-learning layer.

Each structural event the dissection produces (a leg completing, a VWAP
interaction, a level test) becomes a FEATURE VECTOR describing the market context
at the moment the event occurs. These vectors are the inputs the model learns
from; labels (what happened next) are attached separately in labels.py.

THE CAUSALITY RULE (non-negotiable):
Every feature is computable from information available up to and including the
event bar -- never after. A feature that peeks at future bars would leak the
label into the input and produce a model that looks brilliant in backtest and
fails live. Each feature here is annotated with why it is causal. The split
between features (past-only) and labels (future) is the entire foundation of an
honest learning system; if it blurs, nothing downstream is trustworthy.

The unit is ONE EVENT. A session yields many events; a lookback window yields
many sessions; the scanner accumulates all of them into the training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..session.dissect import (
    ClassifiedLeg,
    LegRole,
    SessionDissection,
    VwapEventType,
)
from ..session.session import Session

# Event-type taxonomy for the learning layer. Each is a moment where a decision
# could be made (enter, exit, fade, scalp), so each is worth learning from.
EVENT_TYPES = (
    "leg_complete",  # a directional leg just finished (potential reversal pt)
    "vwap_reclaim",  # price reclaimed VWAP from below
    "vwap_loss",  # price lost VWAP from above
    "vwap_retest_hold",  # retested VWAP and held
    "vwap_retest_fail",  # retested VWAP and failed
    "level_test",  # a structural level was tested
)


@dataclass(frozen=True)
class EventFeatures:
    """
    A single event's causal feature vector.

    `event_index` is the bar at which the event is known (decision bar). All
    numeric features are derived only from bars <= event_index. `as_dict` returns
    a flat, model-ready mapping.
    """

    symbol: str
    date: pd.Timestamp
    event_type: str
    event_index: int
    event_time: pd.Timestamp
    event_price: float

    features: dict[str, float]  # the causal feature vector

    def as_dict(self) -> dict[str, object]:
        base = {
            "symbol": self.symbol,
            "date": self.date,
            "event_type": self.event_type,
            "event_index": self.event_index,
            "event_time": self.event_time,
            "event_price": self.event_price,
        }
        base.update({f"f_{k}": v for k, v in self.features.items()})
        return base


def _causal_atr(session: Session, idx: int) -> float:
    """
    Mean true range using ONLY bars[0:idx+1] (causal).

    session.atr_mean averages true range over the WHOLE session, so normalizing
    a feature at bar `idx` by it injects future volatility into a past feature --
    a lookahead leak the causality test catches. This computes ATR from bars up
    to and including `idx` only. Floored to a small positive to avoid divide-by-
    zero on the first bars.
    """
    b = session.bars.iloc[: idx + 1]
    if len(b) < 2:
        # Not enough bars for true range; use the single bar's range.
        hi = float(b["high"].iloc[0])
        lo = float(b["low"].iloc[0])
        return max(hi - lo, 1e-6)
    high = b["high"].to_numpy(dtype=float)
    low = b["low"].to_numpy(dtype=float)
    prev_close = b["close"].shift(1).fillna(b["close"]).to_numpy(dtype=float)
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return max(float(np.mean(tr)), 1e-6)


def _session_context(session: Session, idx: int) -> dict[str, float]:
    """
    Market context at bar `idx`, computed only from bars[0:idx+1] (causal).

    Captures where price sits relative to the day's developing structure: how far
    into the session, position within the running range, distance from VWAP, the
    pace of the recent move, and realized volatility so far. These are the
    conditions a discretionary trader reads off the tape before acting.

    Normalization uses CAUSAL ATR (up to idx), not the full-session ATR, so no
    future volatility leaks into the feature.
    """
    b = session.bars
    atr = _causal_atr(session, idx)
    upto = b.iloc[: idx + 1]
    close = float(upto["close"].iloc[-1])

    run_high = float(upto["high"].max())
    run_low = float(upto["low"].min())
    run_range = run_high - run_low
    open_px = float(b["open"].iloc[0])

    vwap_series = session.vwap.iloc[: idx + 1]
    vwap_now = float(vwap_series.iloc[-1])

    range_pos = (close - run_low) / run_range if run_range > 0 else 0.5
    frac_elapsed = (idx + 1) / len(b)

    look = min(6, idx + 1)
    recent = float(close - upto["close"].iloc[-look])
    recent_atr = recent / atr

    if idx >= 1:
        c2c = np.diff(upto["close"].to_numpy(dtype=float))
        realized = float(np.std(c2c)) / atr if c2c.size else 0.0
    else:
        realized = 0.0

    return {
        "dist_from_vwap_atr": (close - vwap_now) / atr,
        "range_pos": range_pos,
        "dist_from_open_atr": (close - open_px) / atr,
        "dist_from_runhigh_atr": (close - run_high) / atr,
        "dist_from_runlow_atr": (close - run_low) / atr,
        "frac_elapsed": frac_elapsed,
        "recent_pace_atr": recent_atr,
        "realized_vol_atr": realized,
        "run_range_atr": run_range / atr,
    }


def _leg_features(cl: ClassifiedLeg, atr: float) -> dict[str, float]:
    """Structural features of a just-completed leg (causal: leg is finished)."""
    leg = cl.leg
    return {
        "leg_magnitude_atr": cl.magnitude_atr,
        "leg_bars": float(leg.bars),
        "leg_is_up": 1.0 if leg.direction == "up" else 0.0,
        "leg_crossed_vwap": 1.0 if cl.crossed_vwap else 0.0,
        "leg_reached_hod": 1.0 if cl.reached_hod else 0.0,
        "leg_reached_lod": 1.0 if cl.reached_lod else 0.0,
        "role_flush": 1.0 if LegRole.FLUSH in cl.roles else 0.0,
        "role_reversal": 1.0 if LegRole.VWAP_RECLAIM in cl.roles else 0.0,
        "role_trend": 1.0 if LegRole.TREND_LEG in cl.roles else 0.0,
        "role_retrace": 1.0 if LegRole.RETRACE in cl.roles else 0.0,
        "role_hod_test": 1.0 if LegRole.HOD_TEST in cl.roles else 0.0,
        "role_lod_test": 1.0 if LegRole.LOD_TEST in cl.roles else 0.0,
        "start_below_vwap": 1.0 if cl.start_vs_vwap == "below" else 0.0,
        "end_above_vwap": 1.0 if cl.end_vs_vwap == "above" else 0.0,
    }


def extract_session_events(session: Session, dissection: SessionDissection) -> list[EventFeatures]:
    """
    Every learnable event in one session as causal feature vectors.

    Produces one EventFeatures per: completed leg, VWAP interaction, and level
    test. Each carries session context at its decision bar plus event-specific
    structure. Labels are NOT attached here -- see labels.py.
    """
    atr = session.atr_mean if session.atr_mean > 0 else 1.0
    out: list[EventFeatures] = []

    # Leg-completion events (decision bar = leg end).
    for cl in dissection.classified_legs:
        idx = cl.leg.end_index
        ctx = _session_context(session, idx)
        feats = {**ctx, **_leg_features(cl, atr)}
        out.append(
            EventFeatures(
                symbol=session.symbol,
                date=session.date,
                event_type="leg_complete",
                event_index=idx,
                event_time=cl.leg.end_time,
                event_price=cl.leg.end_price,
                features=feats,
            )
        )

    # VWAP interaction events.
    vtype_map = {
        VwapEventType.RECLAIM: "vwap_reclaim",
        VwapEventType.LOSS: "vwap_loss",
        VwapEventType.RETEST_HOLD: "vwap_retest_hold",
        VwapEventType.RETEST_FAIL: "vwap_retest_fail",
    }
    for ev in dissection.vwap_events:
        ctx = _session_context(session, ev.index)
        feats = {**ctx, "event_vwap_dist_atr": (ev.price - ev.vwap) / atr}
        out.append(
            EventFeatures(
                symbol=session.symbol,
                date=session.date,
                event_type=vtype_map.get(ev.type, "vwap_event"),
                event_index=ev.index,
                event_time=ev.time,
                event_price=ev.price,
                features=feats,
            )
        )

    # Level-test events.
    for le in dissection.level_events:
        ctx = _session_context(session, le.index)
        feats = {
            **ctx,
            "level_held": 1.0 if le.held else 0.0,
            "level_dist_atr": (le.price - le.level) / atr,
        }
        out.append(
            EventFeatures(
                symbol=session.symbol,
                date=session.date,
                event_type="level_test",
                event_index=le.index,
                event_time=le.time,
                event_price=le.price,
                features=feats,
            )
        )

    out.sort(key=lambda e: e.event_index)
    return out
