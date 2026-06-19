"""
Long-term position training-set assembly over a weekly-bar lookback.

Same shape as ml.dataset / swing.dataset: weekly window -> causal position events
(with RS vs a benchmark) -> REUSED ml.labels.label_events with a week-sized
bracket. The benchmark window is supplied so relative-strength features can be
computed; pass None to scan without RS.
"""

from __future__ import annotations

import pandas as pd

from ..core.structural_unit import BarWindow
from ..data.client import FMPClient
from ..ml.labels import BracketSpec, label_events
from .features import DEFAULT_POSITION_SCALE, extract_position_events
from .window import align_benchmark, build_weekly_window

# ~2-month position: +2 ATR target / -1 ATR stop, 8 weeks max hold.
POSITION_BRACKET = BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=8, name="position")


def build_position_frame(
    client: FMPClient,
    symbol: str,
    lookback_days: int = 2920,
    bracket: BracketSpec = POSITION_BRACKET,
    scale_atr: float = DEFAULT_POSITION_SCALE,
    directions: tuple[str, ...] = ("long", "short"),
    benchmark: BarWindow | None = None,
) -> pd.DataFrame:
    """Labeled position-event frame for one symbol. Empty if no data/structure."""
    window = build_weekly_window(client, symbol, lookback_days)
    if window is None:
        return pd.DataFrame()
    bench_close = align_benchmark(window, benchmark) if benchmark is not None else None
    events = extract_position_events(window, scale_atr, bench_close)
    if not events:
        return pd.DataFrame()
    rows = label_events(window, events, bracket, directions=directions, horizon=bracket.max_bars)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "event_index"]).reset_index(drop=True)
