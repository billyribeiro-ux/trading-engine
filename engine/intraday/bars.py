"""
Intraday bar acquisition with honest resolution handling.

FMP native intraday endpoints (verified 2026-06-17):
    1min   -> Ultimate only
    5min   -> Premium+
    15min  -> Premium+
    30min  -> Premium+
    1hour  -> Premium+
FMP does NOT offer 2/3/4-min or 2/4-hour natively. We synthesize those by
resampling a finer native bar -- lossless aggregation, not fabrication:
    2,3,4 min  <- resample from 1min   (requires Ultimate)
    2,4 hour   <- resample from 1hour  (Premium+)

This module resolves a requested timeframe to (a) a native fetch, (b) a native
fetch + resample, or (c) GATED -- and tells the caller which, so nothing is
silently faked. A 3-min request on Premium returns a clear gated error naming
the dependency (1-min / Ultimate), never an empty or wrong-resolution frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..data.client import EndpointGated, FMPClient
from ..data.endpoints import Tier


class Timeframe(str, Enum):
    """All user-selectable timeframes. Value is canonical minutes-as-string."""

    M1 = "1min"
    M2 = "2min"
    M3 = "3min"
    M4 = "4min"
    M5 = "5min"
    M15 = "15min"
    M30 = "30min"
    H1 = "1hour"
    H2 = "2hour"
    H4 = "4hour"

    @property
    def minutes(self) -> int:
        table = {
            "1min": 1,
            "2min": 2,
            "3min": 3,
            "4min": 4,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1hour": 60,
            "2hour": 120,
            "4hour": 240,
        }
        return table[self.value]

    @property
    def pandas_rule(self) -> str:
        """Pandas resample rule string."""
        return f"{self.minutes}min"


# Native FMP endpoints keyed by Timeframe. Absent -> must be resampled.
_NATIVE_ENDPOINT: dict[Timeframe, str] = {
    Timeframe.M1: "intraday_1min",
    Timeframe.M5: "intraday_5min",
    Timeframe.M15: "intraday_15min",
    Timeframe.M30: "intraday_30min",
    Timeframe.H1: "intraday_1hour",
}

# For non-native timeframes: (source native timeframe, endpoint to fetch).
_RESAMPLE_SOURCE: dict[Timeframe, Timeframe] = {
    Timeframe.M2: Timeframe.M1,
    Timeframe.M3: Timeframe.M1,
    Timeframe.M4: Timeframe.M1,
    Timeframe.H2: Timeframe.H1,
    Timeframe.H4: Timeframe.H1,
}


class ResolutionGated(EndpointGated):
    """Requested timeframe needs a tier the key doesn't have."""


@dataclass(frozen=True)
class Resolution:
    """How a requested timeframe will actually be served."""

    requested: Timeframe
    native_source: Timeframe  # the bar we actually fetch
    needs_resample: bool
    min_tier: Tier
    note: str


def resolve_timeframe(tf: Timeframe) -> Resolution:
    """Plan how to serve `tf`, independent of the active key."""
    from ..data.endpoints import get_endpoint

    if tf in _NATIVE_ENDPOINT:
        ep = get_endpoint(_NATIVE_ENDPOINT[tf])
        return Resolution(
            requested=tf,
            native_source=tf,
            needs_resample=False,
            min_tier=ep.min_tier,
            note=f"native {tf.value} ({ep.min_tier.name}+)",
        )
    src = _RESAMPLE_SOURCE[tf]
    ep = get_endpoint(_NATIVE_ENDPOINT[src])
    return Resolution(
        requested=tf,
        native_source=src,
        needs_resample=True,
        min_tier=ep.min_tier,
        note=f"resampled from native {src.value} ({ep.min_tier.name}+)",
    )


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate finer bars to coarser. Standard OHLCV rollup on the right edge.

    Assumes df has a tz-naive 'datetime' column (exchange-local) and the usual
    open/high/low/close/volume. Empty buckets (no trading) are dropped.
    """
    if df.empty:
        return df
    work = df.copy()
    time_col = "datetime" if "datetime" in work.columns else "date"
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.set_index(time_col).sort_index()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    out = work.resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.reset_index().rename(columns={time_col: "datetime"})


def fetch_intraday(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe,
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch intraday bars at the requested timeframe, resampling if needed.

    Raises
    ------
    ResolutionGated
        If the active tier cannot reach the native source for this timeframe.
        The message names the dependency (e.g. '3min needs 1min / Ultimate').

    Returns
    -------
    DataFrame with columns: datetime, open, high, low, close, volume
    sorted ascending. A 'date' (session date) column is added for grouping.
    """
    plan = resolve_timeframe(timeframe)
    if client.tier < plan.min_tier:
        raise ResolutionGated(
            __import__("engine.data.endpoints", fromlist=["get_endpoint"]).get_endpoint(
                _NATIVE_ENDPOINT[plan.native_source]
            ),
            client.tier,
        )

    endpoint_key = _NATIVE_ENDPOINT[plan.native_source]
    params: dict[str, str] = {}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    raw = client.fetch(endpoint_key, symbol=symbol, params=params or None)
    if raw.empty:
        return raw

    # FMP intraday returns a 'date' column that is actually a timestamp.
    if "date" in raw.columns and "datetime" not in raw.columns:
        raw = raw.rename(columns={"date": "datetime"})
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values("datetime").reset_index(drop=True)

    if plan.needs_resample:
        raw = _resample(raw, timeframe.pandas_rule)

    raw["date"] = raw["datetime"].dt.normalize()
    keep = ["datetime", "date", "open", "high", "low", "close"]
    if "volume" in raw.columns:
        keep.append("volume")
    return raw[keep].reset_index(drop=True)


def selectable_timeframes(tier: Tier) -> dict[str, dict]:
    """
    Report every timeframe with its availability at the given tier.

    Surfaced before a run so you see exactly what your key reaches and what an
    upgrade would unlock -- the upgrade-vs-vendor decision you flagged.
    """
    out: dict[str, dict] = {}
    for tf in Timeframe:
        plan = resolve_timeframe(tf)
        out[tf.value] = {
            "available": tier >= plan.min_tier,
            "min_tier": plan.min_tier.name,
            "how": plan.note,
        }
    return out
