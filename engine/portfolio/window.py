"""
Weekly bar windows for the long-term portfolio scanner (Scanner #3).

Resamples daily EOD to weekly (W-FRI) OHLCV and wraps it in a BarWindow — the
same StructuralUnit the intraday/swing scanners use, just at a coarser
resolution. The pivot decomposition then finds long-term legs exactly as before.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.structural_unit import BarWindow
from ..data.client import FMPClient

logger = logging.getLogger("engine.portfolio.window")

MIN_WEEKLY_BARS = 60  # ~14 months of weeks; need MA30 + several legs


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    w = pd.DataFrame(
        {
            "open": d["open"].resample("W-FRI").first(),
            "high": d["high"].resample("W-FRI").max(),
            "low": d["low"].resample("W-FRI").min(),
            "close": d["close"].resample("W-FRI").last(),
            "volume": d["volume"].resample("W-FRI").sum() if "volume" in d else np.nan,
        }
    ).dropna(subset=["open", "high", "low", "close"])
    return w.reset_index().rename(columns={"date": "datetime"})


def build_weekly_window(
    client: FMPClient, symbol: str, lookback_days: int = 2920
) -> BarWindow | None:
    """A weekly BarWindow over ~`lookback_days` (default ~8y). None if too thin."""
    try:
        daily = client.fetch("eod_full", symbol=symbol)
    except Exception as exc:
        logger.warning("No daily EOD for %s: %s", symbol, exc)
        return None
    if daily is None or daily.empty:
        return None
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    d = d[d["date"] >= cutoff]
    weekly = _to_weekly(d)
    if len(weekly) < MIN_WEEKLY_BARS:
        return None
    return BarWindow.from_bars(symbol.strip().upper(), weekly, min_bars=MIN_WEEKLY_BARS)


def align_benchmark(unit: BarWindow, benchmark: BarWindow) -> np.ndarray:
    """Benchmark close aligned to `unit`'s weekly dates (forward-filled).

    Returns an array parallel to unit.bars so relative-strength features can read
    the benchmark at the same bar index. Causal: ffill only carries PAST closes.
    """
    bench = benchmark.bars.set_index("datetime")["close"]
    aligned = bench.reindex(unit.bars["datetime"], method="ffill")
    return aligned.to_numpy(dtype=float)
