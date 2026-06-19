"""
Gap analysis engine.

Computes, for a single symbol over a lookback window, the empirical behaviour of
overnight gaps: how often they continue vs. fill, conditioned on size, direction,
prior trend, relative volume, weekday, and earnings proximity -- each reported
with Wilson intervals and Bayesian shrinkage so thin buckets are flagged, not
trusted.

Inputs are daily OHLCV (split/div adjusted) from the data layer. Daily bars
cannot tell us the *intraday path* to a fill (only intraday data can), so the
fill-timing survival analysis here operates at session granularity: a gap is
'filled this session' if the prior close lies within [low, high] of the gap day.
Sub-session timing requires the 5-min layer, wired separately. We state this
limit rather than pretend daily bars resolve intraday sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

from .statistics import (
    ProportionEstimate,
    SurvivalCurve,
    bootstrap_mean_ci,
    estimate_proportion,
    kaplan_meier,
)

# ATR-size tier boundaries, in ATR units of the gap magnitude.
# A gap of 0.5 ATR is small; >2 ATR is a true event. These edges are deliberate
# and documented so buckets are reproducible across symbols.
_SIZE_EDGES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
_SIZE_LABELS: Final[tuple[str, ...]] = (
    "0-0.25 ATR",
    "0.25-0.5 ATR",
    "0.5-1 ATR",
    "1-2 ATR",
    ">2 ATR",
)


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class GapEvent:
    """One classified gap with its measured outcome."""

    date: pd.Timestamp
    direction: Direction
    gap_atr: float  # signed gap size in ATR units
    gap_pct: float  # signed gap as fraction of prior close
    size_tier: str
    prior_trend: str  # 'up' | 'down' | 'flat' (from regime proxy)
    rvol: float  # volume / rolling median volume
    weekday: str
    near_earnings: bool  # earnings within +/-1 session
    continued: bool  # close on gap side of open
    filled_full: bool  # prior close within [low, high] of gap day
    fill_fraction: float  # how much of the gap closed (0..1, capped)
    fwd_return: float  # open->close return, signed to gap direction


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR on daily bars. Used to normalise gap size."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing == EMA with alpha = 1/period.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _prior_trend(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """
    Cheap, robust trend label from MA relationship + slope sign.

    This is a *proxy* for the regime module (which will replace it with
    ADX/efficiency-ratio/Hurst). Labeled 'prior_trend' and computed only from
    data strictly before the gap day -- no lookahead.
    """
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    slope = ma_fast.diff()
    trend = pd.Series("flat", index=close.index, dtype=object)
    up = (ma_fast > ma_slow) & (slope > 0)
    down = (ma_fast < ma_slow) & (slope < 0)
    trend[up] = "up"
    trend[down] = "down"
    return trend


def classify_gaps(
    daily: pd.DataFrame,
    earnings_dates: set[pd.Timestamp] | None = None,
    atr_period: int = 14,
    rvol_window: int = 20,
    min_gap_atr: float = 0.10,
) -> list[GapEvent]:
    """
    Build the list of classified gap events from daily OHLCV.

    Parameters
    ----------
    daily : DataFrame with columns date, open, high, low, close, volume
            (ascending by date; as returned by the data layer's eod_full).
    earnings_dates : set of session dates with earnings, for near_earnings flag.
    min_gap_atr : ignore micro-gaps below this ATR fraction (pure noise).

    Returns events in chronological order. All conditioning features use only
    information available at or before the gap day's open -- no lookahead leak.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily is missing columns: {sorted(missing)}")
    if len(daily) < max(atr_period, rvol_window) + 5:
        raise ValueError(
            f"Need at least {max(atr_period, rvol_window) + 5} sessions; got {len(daily)}."
        )

    df = daily.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    atr = _atr(df, atr_period)
    prev_close = df["close"].shift(1)
    trend = _prior_trend(df["close"])
    med_vol = df["volume"].rolling(rvol_window).median()
    earnings_dates = earnings_dates or set()
    earnings_norm = {pd.Timestamp(d).normalize() for d in earnings_dates}

    events: list[GapEvent] = []
    for i in range(1, len(df)):
        atr_i = atr.iloc[i]
        pc = prev_close.iloc[i]
        if not np.isfinite(atr_i) or atr_i <= 0 or not np.isfinite(pc) or pc <= 0:
            continue

        o = df["open"].iloc[i]
        gap_abs = o - pc
        gap_atr = gap_abs / atr_i
        if abs(gap_atr) < min_gap_atr:
            continue

        direction = Direction.UP if gap_abs > 0 else Direction.DOWN
        size_tier = _SIZE_LABELS[
            int(np.clip(np.digitize(abs(gap_atr), _SIZE_EDGES) - 1, 0, len(_SIZE_LABELS) - 1))
        ]

        c = df["close"].iloc[i]
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]

        continued = (c >= o) if direction is Direction.UP else (c <= o)
        filled_full = lo <= pc <= hi

        # Fraction of the gap closed: distance price retraced toward prior close.
        if direction is Direction.UP:
            retrace = max(0.0, o - lo)
        else:
            retrace = max(0.0, hi - o)
        fill_fraction = float(np.clip(retrace / abs(gap_abs), 0.0, 1.0)) if gap_abs else 0.0

        # Forward return open->close, signed so positive == continuation.
        raw_ret = (c - o) / o
        fwd_return = raw_ret if direction is Direction.UP else -raw_ret

        rvol = (
            float(df["volume"].iloc[i] / med_vol.iloc[i])
            if np.isfinite(med_vol.iloc[i]) and med_vol.iloc[i] > 0
            else float("nan")
        )

        d = df["date"].iloc[i]
        near_earnings = any(abs((d.normalize() - ed).days) <= 1 for ed in earnings_norm)

        events.append(
            GapEvent(
                date=d,
                direction=direction,
                gap_atr=float(gap_atr),
                gap_pct=float(gap_abs / pc),
                size_tier=size_tier,
                prior_trend=str(trend.iloc[i - 1]),  # trend as of prior close
                rvol=rvol,
                weekday=d.day_name(),
                near_earnings=near_earnings,
                continued=bool(continued),
                filled_full=bool(filled_full),
                fill_fraction=fill_fraction,
                fwd_return=float(fwd_return),
            )
        )
    return events


@dataclass(frozen=True)
class BucketStat:
    """Aggregated statistics for one conditioning bucket."""

    label: str
    n: int
    continuation: ProportionEstimate
    full_fill: ProportionEstimate
    expectancy_mean: float
    expectancy_ci: tuple[float, float]


def _events_to_frame(events: list[GapEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": e.date,
                "direction": e.direction.value,
                "gap_atr": e.gap_atr,
                "size_tier": e.size_tier,
                "prior_trend": e.prior_trend,
                "rvol": e.rvol,
                "weekday": e.weekday,
                "near_earnings": e.near_earnings,
                "continued": e.continued,
                "filled_full": e.filled_full,
                "fill_fraction": e.fill_fraction,
                "fwd_return": e.fwd_return,
            }
            for e in events
        ]
    )


def _bucket_stat(
    label: str,
    sub: pd.DataFrame,
    cont_prior: float,
    fill_prior: float,
) -> BucketStat:
    n = len(sub)
    cont = estimate_proportion(int(sub["continued"].sum()), n, cont_prior)
    fill = estimate_proportion(int(sub["filled_full"].sum()), n, fill_prior)
    mean, lo, hi = bootstrap_mean_ci(sub["fwd_return"].to_numpy())
    return BucketStat(
        label=label,
        n=n,
        continuation=cont,
        full_fill=fill,
        expectancy_mean=mean,
        expectancy_ci=(lo, hi),
    )


@dataclass(frozen=True)
class GapReport:
    symbol: str
    n_events: int
    base_continuation: ProportionEstimate
    base_full_fill: ProportionEstimate
    by_direction: dict[str, BucketStat]
    by_size: dict[str, BucketStat]
    by_trend: dict[str, BucketStat]
    by_weekday: dict[str, BucketStat]
    by_earnings: dict[str, BucketStat]
    fill_survival_up: SurvivalCurve
    fill_survival_down: SurvivalCurve


def analyze_gaps(symbol: str, events: list[GapEvent]) -> GapReport:
    """
    Aggregate classified events into a full conditional report.

    Base rates become the Bayesian priors for every sub-bucket, so a thin cell
    is shrunk toward the symbol's own overall behaviour rather than toward 50%
    or toward nothing.
    """
    if not events:
        raise ValueError(f"No gap events for {symbol}; nothing to analyze.")

    df = _events_to_frame(events)
    n = len(df)
    base_cont_rate = df["continued"].mean()
    base_fill_rate = df["filled_full"].mean()

    base_continuation = estimate_proportion(
        int(df["continued"].sum()), n, prior_mean=0.5, prior_strength=10
    )
    base_full_fill = estimate_proportion(
        int(df["filled_full"].sum()), n, prior_mean=0.5, prior_strength=10
    )

    def grouped(col: str) -> dict[str, BucketStat]:
        out: dict[str, BucketStat] = {}
        for key, sub in df.groupby(col):
            out[str(key)] = _bucket_stat(f"{col}={key}", sub, base_cont_rate, base_fill_rate)
        return out

    # Survival: time-to-fill in *sessions*. With daily bars, a gap either fills
    # the same session (duration 1, observed) or we measure how many sessions
    # until the prior close is first touched (right-censored if never within
    # window). We approximate forward fill horizon at 10 sessions.
    surv_up = _fill_survival(df[df["direction"] == "up"], events, Direction.UP)
    surv_down = _fill_survival(df[df["direction"] == "down"], events, Direction.DOWN)

    return GapReport(
        symbol=symbol,
        n_events=n,
        base_continuation=base_continuation,
        base_full_fill=base_full_fill,
        by_direction=grouped("direction"),
        by_size=grouped("size_tier"),
        by_trend=grouped("prior_trend"),
        by_weekday=grouped("weekday"),
        by_earnings={str(k): v for k, v in grouped("near_earnings").items()},
        fill_survival_up=surv_up,
        fill_survival_down=surv_down,
    )


def _fill_survival(
    sub: pd.DataFrame, events: list[GapEvent], direction: Direction
) -> SurvivalCurve:
    """
    Same-session fill survival from daily bars.

    duration = 1 session, observed = filled_full. This collapses to: of gaps in
    this direction, what fraction filled same-session. The KM machinery is here
    so the 5-min layer can later supply true multi-bar durations with no API
    change to the report. With daily data the curve has a single step.
    """
    if sub.empty:
        return kaplan_meier(np.array([]), np.array([]))
    durations = np.ones(len(sub))
    observed = sub["filled_full"].astype(int).to_numpy()
    return kaplan_meier(durations, observed)
