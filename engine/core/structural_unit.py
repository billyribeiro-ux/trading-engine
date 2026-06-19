"""
The StructuralUnit: the horizon-agnostic thing the pipeline dissects.

The engine is built to carry THREE scanners off one pipeline (see the project's
three-scanner plan): intraday (one 9:30-16:00 auction), swing (a rolling daily
window), and long-term portfolio (weekly/monthly). What they share is a stream of
bars with a volatility scale; what differs is the bar resolution, the events, and
the horizon.

`StructuralUnit` is that shared interface. The multi-scale zigzag decomposition in
`session.pivots` already depends only on `.bars`, `.atr_mean`, `.symbol`, `.date`
and `len(unit)` -- so it works on ANY unit, intraday or daily, with zero changes.
`Session` satisfies this protocol structurally; `BarWindow` is the generic
concrete unit the swing/portfolio scanners build over daily/weekly bars.

There is deliberately NO VWAP or opening-range here -- those are intraday concepts
that live on `Session`. Per-scanner FEATURES are pluggable; only the structural
substrate (bars + ATR scale) is universal.
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

_REQUIRED = ("datetime", "open", "high", "low", "close")


@runtime_checkable
class StructuralUnit(Protocol):
    """What the pivot/leg decomposition (and any generic dissector) needs.

    `bars` is an OHLC(V) frame sorted ascending with a `datetime` column;
    `atr_mean` is a single robust volatility scale (mean true range) used as the
    unit of the multi-scale zigzag thresholds.
    """

    symbol: str
    date: pd.Timestamp

    @property
    def bars(self) -> pd.DataFrame: ...

    @property
    def atr_mean(self) -> float: ...

    def __len__(self) -> int: ...


class BarWindow:
    """A generic StructuralUnit over an arbitrary bar series (daily, weekly, …).

    Unlike `Session` it enforces no session boundary -- it is just a clean,
    sorted, deduped bar window with a mean-true-range scale. The swing scanner
    wraps a multi-year daily series in one of these; the pivot decomposition then
    finds swing legs exactly as it finds intraday legs.
    """

    def __init__(self, symbol: str, date: pd.Timestamp, bars: pd.DataFrame) -> None:
        self.symbol = symbol
        self.date = date
        self._bars = bars

    @classmethod
    def from_bars(cls, symbol: str, bars: pd.DataFrame, min_bars: int = 20) -> BarWindow:
        """Build from an OHLC(V) frame. Accepts a `date` or `datetime` column.

        Sorts ascending, dedupes on the timestamp, and stamps the unit's date as
        the LAST bar (the as-of date — the right edge a signal would act on).
        """
        df = bars.copy()
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        missing = set(_REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
        if len(df) < min_bars:
            raise ValueError(f"{symbol}: only {len(df)} bars; need {min_bars}.")
        if "volume" not in df.columns:
            df["volume"] = np.nan
        as_of = pd.Timestamp(df["datetime"].iloc[-1]).normalize()
        return cls(symbol, as_of, df)

    @property
    def bars(self) -> pd.DataFrame:
        return self._bars.copy()

    def __len__(self) -> int:
        return len(self._bars)

    @cached_property
    def atr_mean(self) -> float:
        """Mean true range over the window (same robust scale as Session.atr_mean).

        Floors to the close-to-close std only on degenerate (flat) data; never
        scales with the window range.
        """
        b = self._bars
        prev_close = b["close"].shift(1).fillna(b["close"])
        tr = np.maximum.reduce(
            [
                (b["high"] - b["low"]).to_numpy(dtype=float),
                (b["high"] - prev_close).abs().to_numpy(dtype=float),
                (b["low"] - prev_close).abs().to_numpy(dtype=float),
            ]
        )
        mean_tr = float(np.mean(tr)) if tr.size else float("nan")
        if np.isfinite(mean_tr) and mean_tr > 1e-9:
            return mean_tr
        close = b["close"].to_numpy(dtype=float)
        c2c = float(np.std(np.diff(close))) if close.size > 1 else 0.0
        return c2c if c2c > 1e-9 else float("nan")

    def __repr__(self) -> str:
        return (
            f"BarWindow({self.symbol} {self.date.date()} bars={len(self)} atr={self.atr_mean:.3f})"
        )
