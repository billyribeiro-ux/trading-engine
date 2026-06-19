"""
Single-session dissection runner.

Fetches real intraday bars for one symbol, builds the regular-hours Session for a
chosen date (or the most recent session available), runs the full dissection, and
returns it. This is the bridge from the data layer to the session-dissection
engine -- the thing you run on real TSLA to see the flush / reversal / VWAP /
HOD-test event sequence on the actual tape.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..data.client import EndpointGated, FMPClient
from ..intraday.bars import Timeframe, fetch_intraday
from .dissect import SessionDissection, dissect_session
from .nested import build_nested_structure
from .pivots import decompose
from .session import build_sessions

logger = logging.getLogger("engine.session.runner")


def _load_prior_daily(client: FMPClient, symbol: str) -> pd.DataFrame | None:
    """Daily OHLC for prior-day levels; None if unavailable at this tier."""
    try:
        d = client.fetch("eod_full", symbol=symbol)
        return d if not d.empty else None
    except EndpointGated:
        return None


def dissect_real_session(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe = Timeframe.M1,
    on_date: str | None = None,
    history_days: int = 7,
) -> tuple[object, SessionDissection, list]:
    """
    Dissect one real trading session.

    Parameters
    ----------
    timeframe : bar resolution. 1min gives the finest read (Ultimate tier);
                5min works on Premium and still resolves the major structure.
    on_date   : 'YYYY-MM-DD' to dissect a specific session, or None for the most
                recent session in the fetched window.
    history_days : how many days back to pull (to locate the session + priors).

    Raises ValueError if the requested session has no regular-hours data.
    """
    symbol = symbol.strip().upper()
    to_date = pd.Timestamp.now().normalize()
    from_date = to_date - pd.Timedelta(days=history_days)
    bars = fetch_intraday(
        client,
        symbol,
        timeframe,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=(to_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if bars.empty:
        raise ValueError(f"No intraday data returned for {symbol}.")

    daily = _load_prior_daily(client, symbol)
    sessions = build_sessions(bars, symbol, daily=daily)
    if not sessions:
        raise ValueError(f"No regular-hours sessions built for {symbol}.")

    if on_date is not None:
        target = pd.Timestamp(on_date).normalize()
        # Session dates may be tz-aware (ET); compare on the naive calendar date.
        match = [s for s in sessions if s.date.tz_localize(None).normalize() == target]
        if not match:
            avail = ", ".join(s.date.strftime("%Y-%m-%d") for s in sessions)
            raise ValueError(f"No session for {symbol} on {on_date}. Available: {avail}")
        session = match[0]
    else:
        session = sessions[-1]

    decomposition = decompose(session)
    dissection = dissect_session(session, decomposition=decomposition)
    nested = (
        build_nested_structure(decomposition, dissection.scale_atr)
        if dissection.scale_atr == dissection.scale_atr
        else []
    )  # NaN guard
    return session, dissection, nested
