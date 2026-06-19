"""
Gap analysis orchestration.

Ties the data layer to the gap engine: pulls daily OHLCV and the earnings
calendar for a symbol, classifies gaps, and produces the conditional report.
Keeps all FMP-specific glue here so the gap engine stays pure (testable on
synthetic frames with no network).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ..data.client import EndpointGated, FMPClient
from .analysis import GapReport, analyze_gaps, classify_gaps

logger = logging.getLogger("engine.gaps.runner")


def _load_earnings_dates(client: FMPClient, symbol: str) -> set[pd.Timestamp]:
    """
    Best-effort earnings-date set for the near_earnings flag.

    If the endpoint is gated or empty, we return an empty set and the report
    simply won't condition on earnings -- an honest absence, not a guess.
    """
    try:
        df = client.fetch("earnings_symbol", symbol=symbol)
    except EndpointGated:
        logger.info("Earnings endpoint gated for this tier; skipping catalyst flag.")
        return set()
    if df.empty:
        return set()
    date_col = next((c for c in ("date", "fillingDate") if c in df.columns), None)
    if date_col is None:
        return set()
    return {pd.Timestamp(d).normalize() for d in pd.to_datetime(df[date_col]).dropna()}


def run_gap_analysis(
    client: FMPClient,
    symbol: str,
    lookback_years: float = 10.0,
    min_gap_atr: float = 0.10,
) -> GapReport:
    """
    Full pipeline for one symbol.

    Pulls daily EOD (the swing/long-term backbone), trims to the lookback,
    attaches earnings context where available, and returns the report.
    """
    symbol = symbol.strip().upper()
    daily = client.fetch("eod_full", symbol=symbol)
    if daily.empty:
        raise ValueError(f"No daily price data returned for {symbol}.")

    daily["date"] = pd.to_datetime(daily["date"])
    cutoff = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(lookback_years * 365.25))
    daily = daily[daily["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    if len(daily) < 60:
        raise ValueError(f"Only {len(daily)} sessions in lookback for {symbol}; widen the window.")

    earnings = _load_earnings_dates(client, symbol)
    events = classify_gaps(daily, earnings_dates=earnings, min_gap_atr=min_gap_atr)
    if not events:
        raise ValueError(
            f"No qualifying gaps for {symbol} in the window "
            f"(min_gap_atr={min_gap_atr}). Lower the threshold or widen lookback."
        )
    logger.info(
        "%s: %d sessions, %d gap events over %.1fy.",
        symbol,
        len(daily),
        len(events),
        lookback_years,
    )
    return analyze_gaps(symbol, events)
