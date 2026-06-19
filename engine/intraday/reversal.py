"""
Multi-scenario intraday reversal detector.

Models the structure you described: an origin extreme forms (often the opening
drive), price makes a counter-move to the opposite extreme, then reverses and
travels back to RETEST the origin extreme. The signal is the return trip; the
outcome is how far the retest gets and what happens there.

The detector is an ENSEMBLE of scenario configurations, not a single rule. Each
config is a distinct, independently-scored hypothesis about what a reversal
looks like. Breadth of scenarios raises coverage; each is scored with its own
conditional probability downstream so the ensemble reveals where real edge
concentrates. No single scenario is assumed correct.

Anchor modes (where the origin extreme is):
    OPENING_RANGE  -> extreme within the first `or_minutes` of the session
    FIRST_EXTREME  -> whichever of session-high/low formed first, any time
    ANY_EXTREME    -> evaluate both high-origin and low-origin independently

Trigger modes (what confirms the reversal has begun):
    VWAP_RECLAIM   -> price crosses back through session VWAP after the counter-extreme
    FOLLOW_THROUGH -> N consecutive bars retracing from the counter-extreme
    BOTH           -> require VWAP reclaim AND follow-through (strictest)

Every combination is a scenario. All are emitted; the backtester scores each.
No lookahead: a signal at bar i uses only bars <= i.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np
import pandas as pd

from .features import add_session_extremes, add_session_vwap, session_atr


class Anchor(str, Enum):
    OPENING_RANGE = "opening_range"
    FIRST_EXTREME = "first_extreme"
    ANY_EXTREME = "any_extreme"


class Trigger(str, Enum):
    VWAP_RECLAIM = "vwap_reclaim"
    FOLLOW_THROUGH = "follow_through"
    BOTH = "both"


class ReversalSide(str, Enum):
    BULLISH = "bullish"  # low-of-day reversal, retests session HIGH (origin)
    BEARISH = "bearish"  # high-of-day reversal, retests session LOW (origin)


@dataclass(frozen=True)
class ScenarioConfig:
    """One detection hypothesis. The cartesian product forms the ensemble."""

    anchor: Anchor
    trigger: Trigger
    or_minutes: int = 30  # opening-range window
    follow_through_bars: int = 2  # bars of retracement to confirm
    min_counter_atr: float = 0.5  # counter-move must travel >= this many ATR
    flush_rvol_min: float = 1.2  # min relative volume on the COUNTER-EXTREME
    # (capitulation flush) bar -- the volume comes
    # on the flush, not the VWAP reclaim. Gating
    # the trigger bar instead throws away valid
    # reversals where volume normalised by reclaim.

    @property
    def name(self) -> str:
        return (
            f"{self.anchor.value}|{self.trigger.value}"
            f"|or{self.or_minutes}|ft{self.follow_through_bars}"
        )


@dataclass(frozen=True)
class ReversalSignal:
    """A detected reversal candidate, ready to be scored by the backtester."""

    symbol: str
    session: pd.Timestamp
    side: ReversalSide
    scenario: str
    signal_time: pd.Timestamp  # bar where reversal is confirmed (entry ref)
    signal_index: int  # row index into the session frame
    entry_price: float  # close at signal bar
    origin_extreme: float  # the level the reversal aims to retest
    counter_extreme: float  # the opposite extreme reached before reversing
    vwap_at_signal: float
    atr_at_signal: float
    rvol_at_signal: float
    minutes_from_open: float


def _session_open_time(session_bars: pd.DataFrame) -> pd.Timestamp:
    return session_bars["datetime"].iloc[0]


def _prepare_session(session_bars: pd.DataFrame) -> pd.DataFrame:
    df = session_bars.copy().reset_index(drop=True)
    df = add_session_vwap(df)
    df = add_session_extremes(df)
    open_t = _session_open_time(df)
    df["minutes_from_open"] = (df["datetime"] - open_t).dt.total_seconds() / 60.0
    if "volume" in df.columns:
        # Running session relative volume vs expanding mean.
        exp_mean = df["volume"].expanding().mean().replace(0, np.nan)
        df["rvol"] = df["volume"] / exp_mean
    else:
        df["rvol"] = np.nan
    return df


def _origin_extreme_index(df: pd.DataFrame, side: ReversalSide, cfg: ScenarioConfig) -> int | None:
    """
    Locate the origin-extreme bar per the anchor mode.

    For BULLISH (retest the HIGH), the origin is a session HIGH.
    For BEARISH (retest the LOW),  the origin is a session LOW.
    """
    if cfg.anchor is Anchor.OPENING_RANGE:
        win = df[df["minutes_from_open"] <= cfg.or_minutes]
        if win.empty:
            return None
        if side is ReversalSide.BULLISH:
            return int(win["high"].idxmax())
        return int(win["low"].idxmin())

    if cfg.anchor is Anchor.FIRST_EXTREME:
        hi_idx = int(df["high"].idxmax())
        lo_idx = int(df["low"].idxmin())
        # Origin is the relevant extreme only if it formed before the opposite.
        if side is ReversalSide.BULLISH:
            return hi_idx if hi_idx < lo_idx else None
        return lo_idx if lo_idx < hi_idx else None

    # ANY_EXTREME: origin is simply the session extreme, wherever it formed.
    if side is ReversalSide.BULLISH:
        return int(df["high"].idxmax())
    return int(df["low"].idxmin())


def _detect_side(
    symbol: str,
    df: pd.DataFrame,
    side: ReversalSide,
    cfg: ScenarioConfig,
) -> ReversalSignal | None:
    """
    Detect a single reversal of `side` in one prepared session frame.

    Sequence enforced (all indices strictly ordered, no lookahead at signal):
        origin_idx  -> counter_idx (opposite extreme, after origin)
                    -> trigger bar (reversal confirmed, after counter)
    Returns the first valid signal in the session, or None.
    """
    origin_idx = _origin_extreme_index(df, side, cfg)
    if origin_idx is None:
        return None

    n = len(df)
    after = df.iloc[origin_idx + 1 :]
    if after.empty:
        return None

    # Counter-extreme = opposite extreme reached AFTER the origin.
    if side is ReversalSide.BULLISH:
        counter_idx = int(after["low"].idxmin())  # the low after the high
        origin_level = float(df["high"].iloc[origin_idx])
        counter_level = float(df["low"].loc[counter_idx])
    else:
        counter_idx = int(after["high"].idxmax())  # the high after the low
        origin_level = float(df["low"].iloc[origin_idx])
        counter_level = float(df["high"].loc[counter_idx])

    if counter_idx <= origin_idx:
        return None

    atr_series = session_atr(df)
    atr_c = atr_series.iloc[counter_idx]
    if not np.isfinite(atr_c) or atr_c <= 0:
        return None

    # Counter-move magnitude gate (in ATR).
    counter_travel = abs(origin_level - counter_level) / atr_c
    if counter_travel < cfg.min_counter_atr:
        return None

    # Volume confirmation on the FLUSH (counter-extreme) bar, not the trigger.
    flush_rvol = df["rvol"].iloc[counter_idx]
    if np.isfinite(flush_rvol) and flush_rvol < cfg.flush_rvol_min:
        return None

    # Search bars after the counter-extreme for the reversal trigger.
    for j in range(counter_idx + 1, n):
        bar = df.iloc[j]
        vwap_j = bar["vwap"]
        if not np.isfinite(vwap_j):
            continue

        vwap_reclaim = (
            bar["close"] > vwap_j if side is ReversalSide.BULLISH else bar["close"] < vwap_j
        )

        # Follow-through: `ft` consecutive closes moving toward the origin.
        ft = cfg.follow_through_bars
        if j - counter_idx >= ft:
            window = df["close"].iloc[j - ft + 1 : j + 1].to_numpy()
            if side is ReversalSide.BULLISH:
                follow = bool(np.all(np.diff(window) > 0)) if window.size > 1 else False
                follow = follow and df["close"].iloc[j] > df["close"].iloc[counter_idx]
            else:
                follow = bool(np.all(np.diff(window) < 0)) if window.size > 1 else False
                follow = follow and df["close"].iloc[j] < df["close"].iloc[counter_idx]
        else:
            follow = False

        if cfg.trigger is Trigger.VWAP_RECLAIM:
            confirmed = vwap_reclaim
        elif cfg.trigger is Trigger.FOLLOW_THROUGH:
            confirmed = follow
        else:  # BOTH
            confirmed = vwap_reclaim and follow

        if not confirmed:
            continue

        rvol_j = float(bar["rvol"]) if np.isfinite(bar["rvol"]) else np.nan
        return ReversalSignal(
            symbol=symbol,
            session=pd.Timestamp(df["date"].iloc[0]),
            side=side,
            scenario=cfg.name,
            signal_time=pd.Timestamp(bar["datetime"]),
            signal_index=j,
            entry_price=float(bar["close"]),
            origin_extreme=origin_level,
            counter_extreme=counter_level,
            vwap_at_signal=float(vwap_j),
            atr_at_signal=float(atr_c),
            rvol_at_signal=rvol_j,
            minutes_from_open=float(bar["minutes_from_open"]),
        )
    return None


def detect_session(
    symbol: str,
    session_bars: pd.DataFrame,
    configs: list[ScenarioConfig],
) -> list[ReversalSignal]:
    """Run every scenario config against one session; emit all signals found."""
    if len(session_bars) < 5:
        return []
    df = _prepare_session(session_bars)
    signals: list[ReversalSignal] = []
    for cfg in configs:
        sides = (
            [ReversalSide.BULLISH, ReversalSide.BEARISH]
            if cfg.anchor is Anchor.ANY_EXTREME
            else [ReversalSide.BULLISH, ReversalSide.BEARISH]
        )
        for side in sides:
            sig = _detect_side(symbol, df, side, cfg)
            if sig is not None:
                signals.append(sig)
    return signals


def default_ensemble() -> list[ScenarioConfig]:
    """
    The full scenario grid: every anchor x trigger, plus parameter variants.

    This is the 'many scenarios' ensemble. Each is scored independently by the
    backtester so the output shows which configurations actually carry edge
    rather than assuming any single one does.
    """
    configs: list[ScenarioConfig] = []
    for anchor, trigger in product(Anchor, Trigger):
        for or_min in (15, 30, 60):
            for ft in (2, 3):
                configs.append(
                    ScenarioConfig(
                        anchor=anchor,
                        trigger=trigger,
                        or_minutes=or_min,
                        follow_through_bars=ft,
                    )
                )
    return configs
