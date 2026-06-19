"""
Session regime classification for edge conditioning.

A reversal that retests the session high 70% of the time on RANGE days and 35% on
strong TREND days has no single "edge" -- it has two, and averaging them hides
both. Institutional systems condition the edge on regime so the live scanner can
say "this setup, in today's regime, has THIS measured edge" rather than a blended
number that is wrong in every regime.

Each session is labeled on two independent axes, using only intraday data from
that session (and prior daily context where noted -- no lookahead beyond the
information available as the session unfolds):

    DIRECTIONAL REGIME (how trending vs mean-reverting the session is):
        TREND_UP / TREND_DOWN / RANGE
        Measured by the intraday efficiency ratio (net move / path length) and
        the sign/slope of a session linear regression. Efficiency near 1 = clean
        trend; near 0 = chop/range.

    VOLATILITY REGIME (how wide the session is vs the instrument's normal):
        HIGH_VOL / NORMAL_VOL / LOW_VOL
        Measured by session range in ATR units vs a rolling baseline.

The classifier is deliberately simple and robust -- efficiency ratio, regression
slope, ATR-relative range -- all real, computable, non-narrative measures. No
chart-pattern storytelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class DirectionalRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"


class VolatilityRegime(str, Enum):
    HIGH_VOL = "high_vol"
    NORMAL_VOL = "normal_vol"
    LOW_VOL = "low_vol"


@dataclass(frozen=True)
class SessionRegime:
    session: pd.Timestamp
    directional: DirectionalRegime
    volatility: VolatilityRegime
    efficiency_ratio: float  # 0..1, net/path
    regression_slope_atr: float  # slope per bar in ATR units (signed)
    range_atr: float  # session high-low in ATR units
    r_squared: float  # linear fit quality

    @property
    def label(self) -> str:
        return f"{self.directional.value}/{self.volatility.value}"


def efficiency_ratio(close: np.ndarray) -> float:
    """
    Kaufman efficiency ratio: |net change| / sum(|bar-to-bar changes|).

    1.0 = perfectly directional; ~0 = pure noise/chop. The clean, standard
    measure of trend vs range, no parameters to overfit.
    """
    if close.size < 2:
        return 0.0
    net = abs(close[-1] - close[0])
    path = np.abs(np.diff(close)).sum()
    return float(net / path) if path > 0 else 0.0


def _regression(close: np.ndarray) -> tuple[float, float]:
    """OLS slope (per bar) and R^2 of close vs bar index."""
    n = close.size
    if n < 3:
        return (0.0, 0.0)
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    y_c = close - close.mean()
    denom = (x_c**2).sum()
    if denom == 0:
        return (0.0, 0.0)
    slope = float((x_c * y_c).sum() / denom)
    y_hat = slope * x_c
    ss_res = ((y_c - y_hat) ** 2).sum()
    ss_tot = (y_c**2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return (slope, r2)


def classify_session_regime(
    session_bars: pd.DataFrame,
    session_atr: float,
    vol_baseline_atr_range: float,
    trend_eff_threshold: float = 0.45,
    trend_slope_atr_threshold: float = 0.02,
    high_vol_mult: float = 1.5,
    low_vol_mult: float = 0.6,
) -> SessionRegime:
    """
    Classify one session's directional and volatility regime.

    Parameters
    ----------
    session_atr : the ATR (price units) used to normalise slope and range.
    vol_baseline_atr_range : the typical session range in ATR units for this
        instrument (rolling median across recent sessions), the denominator for
        the volatility regime. Passing the instrument's own baseline is what
        makes HIGH/LOW vol meaningful rather than absolute.
    """
    close = session_bars["close"].to_numpy(dtype=float)
    high = session_bars["high"].to_numpy(dtype=float)
    low = session_bars["low"].to_numpy(dtype=float)
    session_ts = pd.Timestamp(session_bars["date"].iloc[0])

    eff = efficiency_ratio(close)
    slope, r2 = _regression(close)
    slope_atr = slope / session_atr if session_atr > 0 else 0.0
    range_atr = (high.max() - low.min()) / session_atr if session_atr > 0 else 0.0

    # Directional: needs BOTH efficiency and slope magnitude to call a trend,
    # so a wide-but-choppy day is RANGE, not a false trend.
    is_trend = eff >= trend_eff_threshold and abs(slope_atr) >= trend_slope_atr_threshold
    if is_trend and slope_atr > 0:
        directional = DirectionalRegime.TREND_UP
    elif is_trend and slope_atr < 0:
        directional = DirectionalRegime.TREND_DOWN
    else:
        directional = DirectionalRegime.RANGE

    # Volatility: session range vs the instrument's own baseline.
    if vol_baseline_atr_range > 0:
        ratio = range_atr / vol_baseline_atr_range
    else:
        ratio = 1.0
    if ratio >= high_vol_mult:
        volatility = VolatilityRegime.HIGH_VOL
    elif ratio <= low_vol_mult:
        volatility = VolatilityRegime.LOW_VOL
    else:
        volatility = VolatilityRegime.NORMAL_VOL

    return SessionRegime(
        session=session_ts,
        directional=directional,
        volatility=volatility,
        efficiency_ratio=eff,
        regression_slope_atr=slope_atr,
        range_atr=range_atr,
        r_squared=r2,
    )


def classify_all_sessions(
    bars: pd.DataFrame,
    atr_lookback_sessions: int = 14,
) -> dict[pd.Timestamp, SessionRegime]:
    """
    Classify every session in a multi-session intraday frame.

    Computes each session's ATR proxy (mean true range of its bars) and a rolling
    baseline of session range (in ATR units) from PRIOR sessions only, so the
    volatility regime for session t uses sessions < t -- no lookahead.
    """
    out: dict[pd.Timestamp, SessionRegime] = {}
    sessions = sorted(bars["date"].unique())
    session_range_atr_history: list[float] = []

    for s in sessions:
        sb = bars[bars["date"] == s]
        if len(sb) < 5:
            continue
        high = sb["high"].to_numpy(dtype=float)
        low = sb["low"].to_numpy(dtype=float)
        close = sb["close"].to_numpy(dtype=float)
        # Session ATR proxy: mean bar true range.
        prev_close = np.concatenate([[close[0]], close[:-1]])
        tr = np.maximum.reduce(
            [
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ]
        )
        sess_atr = float(np.mean(tr)) if tr.size else 0.0
        if sess_atr <= 0:
            continue

        # Baseline from prior sessions' range-in-ATR (rolling median).
        if session_range_atr_history:
            baseline = float(np.median(session_range_atr_history[-atr_lookback_sessions:]))
        else:
            baseline = (high.max() - low.min()) / sess_atr  # bootstrap on first

        reg = classify_session_regime(sb, sess_atr, baseline)
        out[pd.Timestamp(s)] = reg
        session_range_atr_history.append(reg.range_atr)

    return out
