"""
Training-set assembly across a lookback window.

The scanner's lookback control says "learn from the last N sessions." This module
turns that into a single labeled, time-ordered dataset: for each session in the
window it runs the dissection, extracts causal features, attaches forward labels,
and concatenates the rows -- tagged with the session date so the walk-forward
split can respect time ordering and purge across session boundaries.

Time ordering is preserved end-to-end. The walk-forward harness (validate.py)
relies on rows being sortable by (date, event_index) so that training always
precedes testing in wall-clock time -- the only split that doesn't cheat.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..data.client import FMPClient
from ..intraday.bars import Timeframe, fetch_intraday
from ..session.dissect import dissect_session
from ..session.pivots import decompose
from ..session.session import build_sessions
from .features import extract_session_events
from .labels import (
    BracketSpec,
    brackets_for_timeframe,
    label_events,
)

logger = logging.getLogger("engine.ml.dataset")


def build_training_frame(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe = Timeframe.M5,
    lookback_days: int = 60,
    bracket: BracketSpec | None = None,
    style: str = "reversal",
    directions: tuple[str, ...] = ("long", "short"),
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Assemble the labeled training set for one symbol over a lookback window.

    Fetches intraday bars across `lookback_days`, splits into regular-hours
    sessions, and for each session produces labeled event rows. Returns a single
    time-ordered DataFrame (sorted by date then event_index). Empty frame if no
    sessions are available.

    bracket/horizon default to TIMEFRAME-AWARE presets (brackets_for_timeframe):
    `style` selects 'reversal' (wide, ~2h) or 'scalp' (tight, ~25m). Pass an
    explicit `bracket`/`horizon` to override. This is what makes 1min and 5min
    both behave sensibly without the caller hand-tuning bar counts.

    The label looks forward up to the bracket's max_bars / the horizon, so events
    too close to a session's end have truncated outcomes -- that is correct and
    honest (a real trade late in the day also has less room), not a bug.
    """
    tf_minutes = Timeframe(timeframe).minutes
    presets = brackets_for_timeframe(tf_minutes)
    if bracket is None:
        bracket = presets.get(style, presets["reversal"])
    if horizon is None:
        horizon = presets["horizon_bars"]
    symbol = symbol.strip().upper()
    to_date = pd.Timestamp.now().normalize()
    from_date = to_date - pd.Timedelta(days=lookback_days)
    bars = fetch_intraday(
        client,
        symbol,
        timeframe,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=(to_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if bars.empty:
        logger.warning("No intraday data for %s in lookback window.", symbol)
        return pd.DataFrame()

    try:
        daily = client.fetch("eod_full", symbol=symbol)
        daily = daily if not daily.empty else None
    except Exception:
        daily = None

    sessions = build_sessions(bars, symbol, daily=daily)
    if not sessions:
        return pd.DataFrame()

    all_rows: list[dict[str, object]] = []
    for session in sessions:
        try:
            dec = decompose(session)
            dissection = dissect_session(session, decomposition=dec)
            events = extract_session_events(session, dissection)
            rows = label_events(session, events, bracket, directions=directions, horizon=horizon)
            all_rows.extend(rows)
        except Exception as exc:  # one bad session shouldn't kill the run
            logger.warning("Skipped %s on %s: %s", symbol, session.date, exc)
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["date", "event_index"]).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns (prefixed f_)."""
    return [c for c in df.columns if c.startswith("f_")]


def label_column(df: pd.DataFrame, kind: str = "win") -> str:
    """Resolve the label column name for a learning target."""
    mapping = {
        "win": "y_win",  # binary: bracket profitable
        "r": "y_bracket_r",  # continuous: realized R
        "return": "y_horizon_return_atr",  # continuous: horizon return
        "target_hit": "y_target_hit",
    }
    col = mapping.get(kind, "y_win")
    if col not in df.columns:
        raise KeyError(f"label '{kind}' -> column '{col}' not in frame")
    return col
