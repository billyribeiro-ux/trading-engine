"""
Swing training-set assembly over a daily-bar lookback.

Mirrors ml.dataset for the swing horizon: fetch daily EOD, wrap a multi-year
window in a BarWindow, extract causal swing events, attach forward bracket labels
(REUSING ml.labels.label_events unchanged — it is unit-agnostic), and return one
time-ordered labeled frame. The label bracket is sized in DAYS, not minutes.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..core.structural_unit import BarWindow
from ..data.client import FMPClient
from ..ml.labels import BracketSpec, label_events
from .features import DEFAULT_SWING_SCALE, extract_swing_events

logger = logging.getLogger("engine.swing.dataset")

# ~2-week swing: +2 ATR target / -1 ATR stop, 10 trading days max hold.
SWING_BRACKET = BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=10, name="swing")

MIN_BARS = 60  # need enough daily history for MA50 + a few swings


def _recent_daily(client: FMPClient, symbol: str, lookback_days: int) -> pd.DataFrame | None:
    try:
        daily = client.fetch("eod_full", symbol=symbol)
    except Exception as exc:  # gated tier / network
        logger.warning("No daily EOD for %s: %s", symbol, exc)
        return None
    if daily is None or daily.empty:
        return None
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    d = d[d["date"] >= cutoff].sort_values("date").reset_index(drop=True)
    return d


def build_swing_window(
    client: FMPClient, symbol: str, lookback_days: int = 730
) -> BarWindow | None:
    """A daily BarWindow over the last `lookback_days`. None if too thin."""
    d = _recent_daily(client, symbol, lookback_days)
    if d is None or len(d) < MIN_BARS:
        return None
    return BarWindow.from_bars(symbol.strip().upper(), d, min_bars=MIN_BARS)


def build_swing_frame(
    client: FMPClient,
    symbol: str,
    lookback_days: int = 730,
    bracket: BracketSpec = SWING_BRACKET,
    scale_atr: float = DEFAULT_SWING_SCALE,
    directions: tuple[str, ...] = ("long", "short"),
) -> pd.DataFrame:
    """Labeled swing-event frame for one symbol. Empty if no data/structure."""
    window = build_swing_window(client, symbol, lookback_days)
    if window is None:
        return pd.DataFrame()
    events = extract_swing_events(window, scale_atr)
    if not events:
        return pd.DataFrame()
    rows = label_events(window, events, bracket, directions=directions, horizon=bracket.max_bars)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "event_index"]).reset_index(drop=True)
