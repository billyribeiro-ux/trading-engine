"""
The Session: one trading day as a self-contained auction.

Every scanner in the engine treats each lookback day as a complete trading
session -- an auction that opens, develops structure bar by bar, and closes --
not as a single daily OHLC dot. This module is the foundation that makes that
literal: it takes one day's intraday bars, enforces the regular-hours boundary
(09:30-16:00 ET), and exposes the day as a clean auction object with its own
opening reference, session VWAP (anchored at the open, the orange line on a
thinkorswim chart), running and final extremes, and prior-day levels.

Why the boundary matters: FMP intraday data includes pre/post-market prints.
If those bleed into the session, the session high/low and the VWAP anchor are
corrupted -- the 384.7 low, the VWAP reclaim, every level the reversal detector
measures against would be wrong. Regular-hours filtering with correct ET tz
handling (DST-aware via stdlib zoneinfo) is therefore not a detail; it is the
precondition for every measurement downstream.

Sessions are immutable once built. Feature extraction, pivot decomposition, and
all scanners consume Session objects, never raw frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from functools import cached_property
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


@dataclass(frozen=True)
class PriorDay:
    high: float
    low: float
    close: float


class Session:
    """
    One regular-hours trading auction.

    Construct via Session.from_intraday(...) which handles tz + filtering, not
    the bare __init__, so a Session always satisfies its invariants:
      * bars are within [09:30, 16:00) ET, sorted, deduped
      * datetime is tz-aware ET
      * at least `min_bars` bars (else construction raises)
    """

    # Note: no __slots__ here -- cached_property requires __dict__ to memoize.
    # Immutability is enforced by construction (from_intraday only) and the
    # absence of public setters; the bar frame is only exposed as a copy.

    def __init__(
        self,
        symbol: str,
        date: pd.Timestamp,
        bars: pd.DataFrame,
        prior_day: PriorDay | None = None,
    ) -> None:
        self.symbol = symbol
        self.date = date
        self._bars = bars
        self._prior_day = prior_day

    # -- construction -------------------------------------------------------
    @classmethod
    def from_intraday(
        cls,
        symbol: str,
        day_bars: pd.DataFrame,
        prior_day: PriorDay | None = None,
        assume_tz: ZoneInfo = ET,
        min_bars: int = 5,
    ) -> Session:
        """
        Build a Session from one day's intraday bars.

        day_bars must have: datetime, open, high, low, close, volume.
        datetime may be tz-naive (assumed `assume_tz`, default ET) or tz-aware
        (converted to ET). Bars outside regular hours are dropped.
        """
        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(day_bars.columns)
        if missing:
            raise ValueError(f"day_bars missing columns: {sorted(missing)}")

        df = day_bars.copy()
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(assume_tz)
        else:
            dt = dt.dt.tz_convert(ET)
        df["datetime"] = dt

        # Regular-hours filter: [09:30, 16:00). Use the time component in ET.
        t = df["datetime"].dt.time
        mask = (t >= REGULAR_OPEN) & (t < REGULAR_CLOSE)
        df = df.loc[mask].copy()

        if df.empty:
            raise ValueError(f"{symbol}: no regular-hours bars on this day after filtering.")

        df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
        if len(df) < min_bars:
            raise ValueError(f"{symbol}: only {len(df)} regular-hours bars; need {min_bars}.")

        if "volume" not in df.columns:
            df["volume"] = np.nan

        session_date = pd.Timestamp(df["datetime"].iloc[0]).normalize()
        return cls(symbol, session_date, df, prior_day)

    # -- raw access ---------------------------------------------------------
    @property
    def bars(self) -> pd.DataFrame:
        """The regular-hours bar frame (a defensive copy)."""
        return self._bars.copy()

    def __len__(self) -> int:
        return len(self._bars)

    # -- opening reference --------------------------------------------------
    @cached_property
    def open_price(self) -> float:
        return float(self._bars["open"].iloc[0])

    @cached_property
    def open_time(self) -> pd.Timestamp:
        return pd.Timestamp(self._bars["datetime"].iloc[0])

    def opening_range(self, minutes: int = 30) -> tuple[float, float]:
        """(high, low) of the first `minutes` of the session."""
        cutoff = self.open_time + pd.Timedelta(minutes=minutes)
        win = self._bars[self._bars["datetime"] < cutoff]
        if win.empty:
            win = self._bars.iloc[:1]
        return (float(win["high"].max()), float(win["low"].min()))

    # -- extremes -----------------------------------------------------------
    @cached_property
    def high(self) -> float:
        return float(self._bars["high"].max())

    @cached_property
    def low(self) -> float:
        return float(self._bars["low"].min())

    @cached_property
    def high_time(self) -> pd.Timestamp:
        return pd.Timestamp(self._bars.loc[self._bars["high"].idxmax(), "datetime"])

    @cached_property
    def low_time(self) -> pd.Timestamp:
        return pd.Timestamp(self._bars.loc[self._bars["low"].idxmin(), "datetime"])

    @cached_property
    def close_price(self) -> float:
        return float(self._bars["close"].iloc[-1])

    # -- session VWAP (the orange line) ------------------------------------
    @cached_property
    def vwap(self) -> pd.Series:
        """
        Running session VWAP anchored at the open, volume-weighted on typical
        price. Index-aligned to bars. If volume is missing, falls back to a
        running mean of typical price (flagged by vwap_is_volume_weighted).
        """
        b = self._bars
        typical = (b["high"] + b["low"] + b["close"]) / 3.0
        if self.vwap_is_volume_weighted:
            vol = b["volume"].fillna(0).clip(lower=0)
            cum_pv = (typical * vol).cumsum()
            cum_v = vol.cumsum().replace(0, np.nan)
            v = cum_pv / cum_v
        else:
            v = typical.expanding().mean()
        return v.ffill().reset_index(drop=True)

    @cached_property
    def vwap_is_volume_weighted(self) -> bool:
        vol = self._bars["volume"]
        return bool(vol.notna().any() and vol.fillna(0).sum() > 0)

    @cached_property
    def vwap_final(self) -> float:
        return float(self.vwap.iloc[-1])

    # -- running extremes (no lookahead) -----------------------------------
    @cached_property
    def running_high(self) -> pd.Series:
        return self._bars["high"].cummax().reset_index(drop=True)

    @cached_property
    def running_low(self) -> pd.Series:
        return self._bars["low"].cummin().reset_index(drop=True)

    # -- ATR ----------------------------------------------------------------
    def atr(self, period: int = 14) -> pd.Series:
        b = self._bars
        prev_close = b["close"].shift(1)
        tr = pd.concat(
            [b["high"] - b["low"], (b["high"] - prev_close).abs(), (b["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return (
            tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period)
            .mean()
            .reset_index(drop=True)
        )

    @cached_property
    def atr_mean(self) -> float:
        """
        Robust single-number session ATR (mean bar true range).

        Mean true range is the correct measure. We floor it only against a
        genuine collapse toward zero (e.g. perfectly smooth synthetic data),
        using the std of close-to-close changes as the floor -- NOT against the
        session range. An earlier version floored with range/sqrt(bars), which
        scales with the range and therefore forces range/atr ~= sqrt(n_bars) on
        every session, masking the true ATR. That was wrong; mean TR stands on
        its own for any session with real intrabar movement.
        """
        b = self._bars
        prev_close = b["close"].shift(1).fillna(b["close"])
        tr = np.maximum.reduce(
            [
                (b["high"] - b["low"]).to_numpy(),
                (b["high"] - prev_close).abs().to_numpy(),
                (b["low"] - prev_close).abs().to_numpy(),
            ]
        )
        mean_tr = float(np.mean(tr)) if tr.size else float("nan")

        # Only floor if mean TR is degenerate (near zero). The floor is the
        # close-to-close volatility, which captures drift the TR may miss on
        # pathological flat data. It does NOT scale with the session range.
        close = b["close"].to_numpy(dtype=float)
        c2c_std = float(np.std(np.diff(close))) if close.size > 1 else 0.0

        if np.isfinite(mean_tr) and mean_tr > 1e-9:
            return mean_tr
        return c2c_std if c2c_std > 1e-9 else float("nan")

    # -- prior-day levels ---------------------------------------------------
    @property
    def prior_day(self) -> PriorDay | None:
        return self._prior_day

    # -- minutes-from-open helper ------------------------------------------
    @cached_property
    def minutes_from_open(self) -> pd.Series:
        return ((self._bars["datetime"] - self.open_time).dt.total_seconds() / 60.0).reset_index(
            drop=True
        )

    def __repr__(self) -> str:
        return (
            f"Session({self.symbol} {self.date.date()} "
            f"bars={len(self)} O={self.open_price:.2f} H={self.high:.2f} "
            f"L={self.low:.2f} C={self.close_price:.2f} VWAP_f={self.vwap_final:.2f})"
        )


def build_sessions(
    intraday: pd.DataFrame,
    symbol: str,
    daily: pd.DataFrame | None = None,
    assume_tz: ZoneInfo = ET,
    min_bars: int = 5,
) -> list[Session]:
    """
    Split a multi-day intraday frame into a list of Sessions, one per trading
    day, each a clean regular-hours auction. Optionally attach prior-day levels
    from a daily OHLC frame.

    This is the bridge the lookback uses: N days of intraday -> N Sessions, each
    analyzed start to finish as its own trading day. Days with too few
    regular-hours bars are skipped (with no exception) so one thin day does not
    abort the whole lookback.
    """
    df = intraday.copy()
    dt = pd.to_datetime(df["datetime"])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(assume_tz)
    else:
        dt = dt.dt.tz_convert(ET)
    df["datetime"] = dt
    df["_session_date"] = df["datetime"].dt.normalize()

    prior_lookup: dict[pd.Timestamp, PriorDay] = {}
    if daily is not None and not daily.empty:
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d = d.sort_values("date").reset_index(drop=True)
        for i in range(1, len(d)):
            # ET-normalize the daily date to match session keys.
            key = pd.Timestamp(d["date"].iloc[i]).tz_localize(ET)
            prior_lookup[key] = PriorDay(
                high=float(d["high"].iloc[i - 1]),
                low=float(d["low"].iloc[i - 1]),
                close=float(d["close"].iloc[i - 1]),
            )

    sessions: list[Session] = []
    for sdate, day_bars in df.groupby("_session_date"):
        prior = prior_lookup.get(pd.Timestamp(sdate))
        try:
            sessions.append(
                Session.from_intraday(
                    symbol,
                    day_bars.drop(columns="_session_date"),
                    prior_day=prior,
                    assume_tz=assume_tz,
                    min_bars=min_bars,
                )
            )
        except ValueError:
            continue
    return sessions
