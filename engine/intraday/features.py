"""
Session-anchored intraday features.

Everything here is computed *within* a trading session and uses only bars at or
before the current bar -- no lookahead. These are the reference levels a
reversal is measured against: the running VWAP (the intraday mean a reversal
reverts toward), the running session high/low (the extreme being rejected), and
prior-day levels (PDH/PDL/prior-close) carried in as static targets.

VWAP here is the standard intraday volume-weighted average price, reset each
session and accumulated bar by bar -- identical to what a trading platform draws,
so signals line up with what you'd see on the chart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def add_session_vwap(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add a running, session-reset VWAP column.

    Requires columns: datetime, date, high, low, close, volume.
    If volume is absent (some feeds), VWAP falls back to a running mean of the
    typical price -- flagged via the 'vwap_volume_weighted' bool column so the
    detector knows whether the VWAP is true volume-weighted or a proxy.
    """
    df = bars.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0

    has_vol = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
    if has_vol:
        vol = df["volume"].fillna(0).clip(lower=0)
        pv = typical * vol
        df["_cum_pv"] = pv.groupby(df["date"]).cumsum()
        df["_cum_vol"] = vol.groupby(df["date"]).cumsum().replace(0, np.nan)
        df["vwap"] = df["_cum_pv"] / df["_cum_vol"]
        df = df.drop(columns=["_cum_pv", "_cum_vol"])
        df["vwap_volume_weighted"] = True
    else:
        df["vwap"] = typical.groupby(df["date"]).expanding().mean().reset_index(level=0, drop=True)
        df["vwap_volume_weighted"] = False
    df["vwap"] = df.groupby("date")["vwap"].ffill()
    return df


def add_session_extremes(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add running session high/low *as of each bar* (inclusive of current bar),
    and the same exclusive of the current bar (the extreme the current bar must
    break to make a NEW extreme). Exclusive versions are what the detector tests
    against, so a bar setting a new high is compared to the prior running high.
    """
    df = bars.copy()
    g = df.groupby("date")
    df["sess_high"] = g["high"].cummax()
    df["sess_low"] = g["low"].cummin()
    # Exclusive (prior-bar) running extreme within the session.
    df["sess_high_prev"] = g["high"].cummax().groupby(df["date"]).shift(1)
    df["sess_low_prev"] = g["low"].cummin().groupby(df["date"]).shift(1)
    return df


@dataclass(frozen=True)
class PriorDayLevels:
    pdh: float
    pdl: float
    prev_close: float


def prior_day_levels(daily: pd.DataFrame) -> dict[pd.Timestamp, PriorDayLevels]:
    """
    Map each session date -> prior session's high/low/close.

    `daily` is the daily OHLC frame (from the data layer). Used to attach static
    targets (PDH/PDL/prior-close) to each intraday session. Keyed by the session
    the levels apply *to* (i.e. shifted forward by one trading day).
    """
    d = daily.copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    out: dict[pd.Timestamp, PriorDayLevels] = {}
    for i in range(1, len(d)):
        out[d["date"].iloc[i]] = PriorDayLevels(
            pdh=float(d["high"].iloc[i - 1]),
            pdl=float(d["low"].iloc[i - 1]),
            prev_close=float(d["close"].iloc[i - 1]),
        )
    return out


def session_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Intraday ATR on the bar series (Wilder), for normalising rejection size and
    setting R-multiple targets. Computed across the continuous bar stream; this
    is intentional -- volatility regime carries across sessions.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
