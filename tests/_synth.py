"""
Deterministic synthetic data for the test suite.

The Prime Directive: tests must actively try to BREAK the engine. Synthetic data
here is therefore built with *known* properties so a test can assert an exact,
hand-computable outcome -- not "it ran." Where a test needs realistic structure
(multi-leg sessions for the dissection/consistency layer) the path is randomized
but seeded, so failures reproduce.

IMPORTANT for ATR-based tests: `constant_tr_session` produces bars whose mean
true range is EXACTLY `tr`, so `_causal_atr(session, k)` == `tr` for every k>=1
(all bars share that TR). That lets bracket-math tests place targets/stops at
precise prices and assert exact R outcomes.

This module is a shared dependency of every test file. Do not edit it to make a
single test pass -- fix the test or report an engine bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.session.session import Session

DEFAULT_DATE = "2026-06-01"  # a Monday; arbitrary but fixed
SESSION_OPEN = "09:30"


def make_intraday_df(
    opens,
    highs,
    lows,
    closes,
    volumes=None,
    *,
    date: str = DEFAULT_DATE,
    start: str = SESSION_OPEN,
    freq_min: int = 1,
) -> pd.DataFrame:
    """OHLCV frame with ET-naive datetimes stepping `freq_min` from the open."""
    n = len(closes)
    ts = pd.date_range(f"{date} {start}", periods=n, freq=f"{freq_min}min")
    if volumes is None:
        volumes = np.full(n, 1000.0)
    return pd.DataFrame(
        {
            "datetime": ts,
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.asarray(volumes, dtype=float),
        }
    )


def bars_from_closes(
    closes,
    *,
    wick: float = 0.1,
    vol: float = 1000.0,
    date: str = DEFAULT_DATE,
    freq_min: int = 1,
) -> pd.DataFrame:
    """
    Build a realistic OHLCV frame from a close path.

    open[i] = close[i-1] (continuous tape), high/low extend `wick` beyond the
    open/close envelope. Volume is constant so VWAP is genuinely volume-weighted.
    """
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    highs = body_hi + wick
    lows = body_lo - wick
    vols = np.full(n, vol)
    return make_intraday_df(opens, highs, lows, closes, vols, date=date, freq_min=freq_min)


def session_from_closes(closes, *, symbol: str = "TEST", prior_day=None, **kw) -> Session:
    """A Session built from a close path (see bars_from_closes for OHLC rules)."""
    df = bars_from_closes(closes, **kw)
    return Session.from_intraday(symbol, df, prior_day=prior_day)


def constant_tr_session(
    n: int,
    *,
    base: float = 100.0,
    tr: float = 1.0,
    symbol: str = "TEST",
    prior_day=None,
) -> Session:
    """
    `n` bars whose every true range is EXACTLY `tr` and whose closes are flat at
    `base`. Therefore mean-TR == `tr` and _causal_atr(session, k) == `tr` for all
    k>=1. Open==close==base; high=base+tr/2; low=base-tr/2.
    """
    base_arr = np.full(n, base)
    highs = base_arr + tr / 2.0
    lows = base_arr - tr / 2.0
    df = make_intraday_df(base_arr, highs, lows, base_arr)
    return Session.from_intraday(symbol, df, prior_day=prior_day)


def append_path_session(
    pre_n: int,
    path_bars: list,
    *,
    base: float = 100.0,
    tr: float = 1.0,
    symbol: str = "TEST",
) -> Session:
    """
    A session with `pre_n` constant-TR bars (so causal ATR at index pre_n-1 == tr),
    followed by explicit `path_bars` rows. Each path bar is a dict with keys
    open/high/low/close (volume optional). Use this to plant a KNOWN forward path
    after an event bar for bracket-label tests.
    """
    base_arr = np.full(pre_n, base)
    o = list(base_arr)
    h = list(base_arr + tr / 2.0)
    low_list = list(base_arr - tr / 2.0)
    c = list(base_arr)
    v = [1000.0] * pre_n
    for pb in path_bars:
        o.append(float(pb["open"]))
        h.append(float(pb["high"]))
        low_list.append(float(pb["low"]))
        c.append(float(pb["close"]))
        v.append(float(pb.get("volume", 1000.0)))
    df = make_intraday_df(o, h, low_list, c, v)
    return Session.from_intraday(symbol, df)


def multileg_closes(
    rng: np.random.Generator,
    *,
    n_legs: int = 6,
    bars_per_leg: int = 12,
    step: float = 0.4,
    noise: float = 0.02,
    start: float = 100.0,
) -> np.ndarray:
    """
    A clean alternating multi-swing close path. Each leg is several ATR (slope
    dominates the per-bar wick/noise), so a mid-range scale resolves ~n_legs
    distinct legs and `primary_scale` finds real structure (>=4 merged legs).
    """
    closes = [float(start)]
    direction = 1
    for _ in range(n_legs):
        bpl = max(5, bars_per_leg + int(rng.integers(-3, 4)))
        slope = direction * (step + float(rng.uniform(0.0, 0.2)))
        for _ in range(bpl):
            closes.append(closes[-1] + slope + float(rng.normal(0.0, noise)))
        direction *= -1
    return np.asarray(closes, dtype=float)


def multileg_session(rng: np.random.Generator, *, symbol: str = "TEST", **kw) -> Session:
    """A realistic multi-leg Session for the structure/dissection/consistency layer."""
    return session_from_closes(multileg_closes(rng, **kw), symbol=symbol)


def corrupt_future(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """
    Return a copy of `df` with every bar AFTER `idx` replaced by garbage (prices
    blown up, lows driven negative-ish), preserving the datetime column so the
    Session still builds. Bars 0..idx are untouched. This is the causality probe:
    any past feature that changes when the future is corrupted is a lookahead leak.
    """
    out = df.copy().reset_index(drop=True)
    fut = out.index > idx
    out.loc[fut, "high"] = out.loc[fut, "high"] * 7.0 + 500.0
    out.loc[fut, "low"] = out.loc[fut, "low"] * 0.01 - 50.0
    out.loc[fut, "open"] = out.loc[fut, "open"] * 3.0 + 100.0
    out.loc[fut, "close"] = out.loc[fut, "close"] * 0.5 - 25.0
    out.loc[fut, "volume"] = out.loc[fut, "volume"] * 13.0 + 1.0
    return out
