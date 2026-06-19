"""
Live reversal evaluation at the right edge of the session.

The backtester asks "did a reversal that fired in the past work?". The live
evaluator asks "what is the state of a reversal structure RIGHT NOW, as of the
last closed bar?". Same detection logic, applied to the session-so-far, with one
non-negotiable rule: the in-progress (forming) bar is DROPPED. Its high/low/close
are not final; evaluating it would produce signals that repaint when the bar
closes. Everything here keys off completed bars only.

Each ticker resolves to one state per scenario:
    CONFIRMED -- trigger satisfied on the last closed bar (or already satisfied
                 and price has not yet reached origin/stop): actionable now
    FORMING   -- counter-extreme in place, trigger partially met (e.g. price
                 approaching VWAP from the wrong side, follow-through building)
    WATCH     -- origin extreme + qualifying counter-move complete, no trigger yet
    NONE      -- no qualifying structure this session

The evaluator returns the live structural state plus the reference levels a
trader needs (entry zone, origin target, stop) so the scan output is directly
actionable, not just a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .features import session_atr
from .reversal import (
    ReversalSide,
    ScenarioConfig,
    Trigger,
    _origin_extreme_index,
    _prepare_session,
)


class SignalState(str, Enum):
    CONFIRMED = "confirmed"
    FORMING = "forming"
    WATCH = "watch"
    NONE = "none"

    @property
    def rank(self) -> int:
        return {"confirmed": 3, "forming": 2, "watch": 1, "none": 0}[self.value]


@dataclass(frozen=True)
class LiveEvaluation:
    """Live state of one scenario on one ticker, as of the last closed bar."""

    symbol: str
    session: pd.Timestamp
    side: ReversalSide
    scenario: str
    state: SignalState
    last_closed_time: pd.Timestamp
    last_price: float
    origin_extreme: float  # target on reversal completion
    counter_extreme: float  # stop reference
    vwap: float
    atr: float
    distance_to_trigger_atr: float  # how far price is from triggering, in ATR
    minutes_from_open: float
    note: str


def drop_forming_bar(
    session_bars: pd.DataFrame, now: pd.Timestamp | None = None, bar_minutes: int | None = None
) -> pd.DataFrame:
    """
    Remove the in-progress bar so only closed bars are evaluated.

    If `now` and `bar_minutes` are given, a final bar whose close time is in the
    future relative to `now` is treated as forming and dropped. Absent those, we
    conservatively drop the last bar only if the feed appended a partial one
    (detected by an irregular final interval). When in doubt we keep data; the
    detector's confirmation requirement already guards against acting on a single
    unstable bar, and the scan refreshes on the next poll.
    """
    if session_bars.empty:
        return session_bars
    df = session_bars.sort_values("datetime").reset_index(drop=True)
    if now is not None and bar_minutes is not None and len(df) >= 1:
        last_open = pd.Timestamp(df["datetime"].iloc[-1])
        expected_close = last_open + pd.Timedelta(minutes=bar_minutes)
        if pd.Timestamp(now) < expected_close:
            return df.iloc[:-1].reset_index(drop=True)
    return df


def evaluate_live(
    symbol: str,
    session_bars: pd.DataFrame,
    cfg: ScenarioConfig,
    now: pd.Timestamp | None = None,
    bar_minutes: int | None = None,
) -> list[LiveEvaluation]:
    """
    Evaluate one scenario's live state for both sides on the session-so-far.

    Returns one LiveEvaluation per side (bullish/bearish) with a non-NONE state,
    or an empty list if no structure qualifies. Closed-bars-only.
    """
    bars = drop_forming_bar(session_bars, now=now, bar_minutes=bar_minutes)
    if len(bars) < 5:
        return []
    df = _prepare_session(bars)
    atr_series = session_atr(df)
    results: list[LiveEvaluation] = []

    for side in (ReversalSide.BULLISH, ReversalSide.BEARISH):
        ev = _evaluate_side(symbol, df, side, cfg, atr_series)
        if ev is not None and ev.state is not SignalState.NONE:
            results.append(ev)
    return results


def _evaluate_side(
    symbol: str,
    df: pd.DataFrame,
    side: ReversalSide,
    cfg: ScenarioConfig,
    atr_series: pd.Series,
) -> LiveEvaluation | None:
    n = len(df)
    last = df.iloc[-1]
    last_price = float(last["close"])
    vwap = float(last["vwap"]) if np.isfinite(last["vwap"]) else last_price
    atr = float(atr_series.iloc[-1]) if np.isfinite(atr_series.iloc[-1]) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None

    origin_idx = _origin_extreme_index(df, side, cfg)
    if origin_idx is None:
        return _none(symbol, df, side, cfg, last, last_price, vwap, atr, "no origin extreme")

    after = df.iloc[origin_idx + 1 :]
    if after.empty:
        return _none(symbol, df, side, cfg, last, last_price, vwap, atr, "origin just formed")

    if side is ReversalSide.BULLISH:
        counter_idx = int(after["low"].idxmin())
        origin_level = float(df["high"].iloc[origin_idx])
        counter_level = float(df["low"].loc[counter_idx])
    else:
        counter_idx = int(after["high"].idxmax())
        origin_level = float(df["low"].iloc[origin_idx])
        counter_level = float(df["high"].loc[counter_idx])

    counter_travel = abs(origin_level - counter_level) / atr
    minutes = float(last["minutes_from_open"])

    # Volume confirmation on the flush bar (same rule as the backtester).
    flush_rvol = df["rvol"].iloc[counter_idx]
    flush_ok = (not np.isfinite(flush_rvol)) or flush_rvol >= cfg.flush_rvol_min

    base_kwargs = dict(
        symbol=symbol,
        session=pd.Timestamp(df["date"].iloc[0]),
        side=side,
        scenario=cfg.name,
        last_closed_time=pd.Timestamp(last["datetime"]),
        last_price=last_price,
        origin_extreme=origin_level,
        counter_extreme=counter_level,
        vwap=vwap,
        atr=atr,
        minutes_from_open=minutes,
    )

    # Counter-move not yet large enough, or flush lacked volume -> nothing.
    if counter_travel < cfg.min_counter_atr or not flush_ok:
        return LiveEvaluation(
            **base_kwargs,
            state=SignalState.NONE,
            distance_to_trigger_atr=float("nan"),
            note=(
                f"counter-move {counter_travel:.2f}ATR < {cfg.min_counter_atr}"
                if counter_travel < cfg.min_counter_atr
                else f"flush rvol {flush_rvol:.2f} < {cfg.flush_rvol_min}"
            ),
        )

    # Has the trigger already fired on a closed bar at/after counter?
    confirmed_idx = _first_trigger_index(df, side, cfg, counter_idx)
    if confirmed_idx is not None:
        # Confirmed if price has neither reached origin nor stop yet (still live),
        # else it's historical for this session.
        reached_origin, hit_stop = _resolved(df, side, confirmed_idx, origin_level, counter_level)
        if not reached_origin and not hit_stop:
            return LiveEvaluation(
                **base_kwargs,
                state=SignalState.CONFIRMED,
                distance_to_trigger_atr=0.0,
                note=f"triggered {n - 1 - confirmed_idx} bars ago, en route to origin",
            )
        # Trigger fired but already resolved this session.
        return LiveEvaluation(
            **base_kwargs,
            state=SignalState.NONE,
            distance_to_trigger_atr=float("nan"),
            note="reversal already resolved this session",
        )

    # No trigger yet: decide FORMING vs WATCH by proximity to the trigger.
    if side is ReversalSide.BULLISH:
        dist_atr = (vwap - last_price) / atr  # need close above vwap
    else:
        dist_atr = (last_price - vwap) / atr  # need close below vwap

    if cfg.trigger in (Trigger.VWAP_RECLAIM, Trigger.BOTH):
        if dist_atr <= 0.25:  # within a quarter-ATR of reclaiming
            state = SignalState.FORMING
            note = f"approaching VWAP reclaim ({dist_atr:.2f}ATR away)"
        else:
            state = SignalState.WATCH
            note = f"counter-move done; {dist_atr:.2f}ATR from VWAP trigger"
    else:
        # Follow-through trigger: forming if last bars already turning.
        turning = _turning(df, side, cfg.follow_through_bars)
        state = SignalState.FORMING if turning else SignalState.WATCH
        note = "follow-through building" if turning else "awaiting follow-through"

    return LiveEvaluation(
        **base_kwargs,
        state=state,
        distance_to_trigger_atr=float(dist_atr),
        note=note,
    )


def _first_trigger_index(
    df: pd.DataFrame, side: ReversalSide, cfg: ScenarioConfig, counter_idx: int
) -> int | None:
    n = len(df)
    for j in range(counter_idx + 1, n):
        bar = df.iloc[j]
        vwap_j = bar["vwap"]
        if not np.isfinite(vwap_j):
            continue
        vwap_reclaim = (
            bar["close"] > vwap_j if side is ReversalSide.BULLISH else bar["close"] < vwap_j
        )
        ft = cfg.follow_through_bars
        follow = False
        if j - counter_idx >= ft:
            window = df["close"].iloc[j - ft + 1 : j + 1].to_numpy()
            if side is ReversalSide.BULLISH:
                follow = (
                    window.size > 1
                    and bool(np.all(np.diff(window) > 0))
                    and df["close"].iloc[j] > df["close"].iloc[counter_idx]
                )
            else:
                follow = (
                    window.size > 1
                    and bool(np.all(np.diff(window) < 0))
                    and df["close"].iloc[j] < df["close"].iloc[counter_idx]
                )
        if cfg.trigger is Trigger.VWAP_RECLAIM:
            ok = vwap_reclaim
        elif cfg.trigger is Trigger.FOLLOW_THROUGH:
            ok = follow
        else:
            ok = vwap_reclaim and follow
        if not ok:
            continue
        return j
    return None


def _resolved(
    df: pd.DataFrame,
    side: ReversalSide,
    trig_idx: int,
    origin: float,
    counter: float,
) -> tuple[bool, bool]:
    fwd = df.iloc[trig_idx + 1 :]
    if fwd.empty:
        return (False, False)
    if side is ReversalSide.BULLISH:
        return (bool((fwd["high"] >= origin).any()), bool((fwd["low"] <= counter).any()))
    return (bool((fwd["low"] <= origin).any()), bool((fwd["high"] >= counter).any()))


def _turning(df: pd.DataFrame, side: ReversalSide, ft: int) -> bool:
    if len(df) < ft + 1:
        return False
    window = df["close"].iloc[-ft:].to_numpy()
    if window.size < 2:
        return False
    return (
        bool(np.all(np.diff(window) > 0))
        if side is ReversalSide.BULLISH
        else bool(np.all(np.diff(window) < 0))
    )


def _none(symbol, df, side, cfg, last, last_price, vwap, atr, note):
    return LiveEvaluation(
        symbol=symbol,
        session=pd.Timestamp(df["date"].iloc[0]),
        side=side,
        scenario=cfg.name,
        state=SignalState.NONE,
        last_closed_time=pd.Timestamp(last["datetime"]),
        last_price=last_price,
        origin_extreme=float("nan"),
        counter_extreme=float("nan"),
        vwap=vwap,
        atr=atr,
        distance_to_trigger_atr=float("nan"),
        minutes_from_open=float(last["minutes_from_open"]),
        note=note,
    )
