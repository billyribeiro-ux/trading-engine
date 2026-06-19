"""
Session dissection: classify every leg and log every VWAP / level interaction.

This is the layer that reads a session back as a sequence of named, measured
events -- the "dissect every move on the chart" engine. It consumes a Session and
its multi-scale decomposition and produces, for the major structural scale:

    LEG CLASSIFICATION -- each directional leg labeled by its role in the auction:
        flush            -- a sharp directional move to a session extreme
        reversal         -- the leg off an extreme that turns the auction
        vwap_reclaim     -- a leg that crosses back through VWAP from below/above
        trend_leg        -- a sustained continuation leg in the dominant direction
        retrace          -- a counter-trend pullback within a trend
        hod_test / lod_test -- a leg that reaches the session high / low
        fade             -- a leg off an extreme that gives back gains into close

    VWAP INTERACTION EVENTS -- every crossing/touch of VWAP with its outcome:
        reclaim (cross from below to above) / loss (above to below)
        retest_hold (approach VWAP and bounce) / retest_fail (approach and break)

    LEVEL EVENTS -- every test of session high, session low, prior-day H/L/close,
        and the opening range, with hold-or-break outcome.

All of this is measured, not narrated: each event carries prices, times, ATR-
normalized magnitudes, and the subsequent outcome. This is the feature substrate
the self-learning layer trains on, and the human-readable dissection of any one
session.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .pivots import (
    Leg,
    MultiScaleDecomposition,
    build_skeleton,
    decompose,
    merge_insignificant_swings,
)
from .session import Session


class LegRole(str, Enum):
    FLUSH = "flush"
    REVERSAL = "reversal"
    VWAP_RECLAIM = "vwap_reclaim"
    VWAP_LOSS = "vwap_loss"
    TREND_LEG = "trend_leg"
    RETRACE = "retrace"
    HOD_TEST = "hod_test"
    LOD_TEST = "lod_test"
    FADE = "fade"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class ClassifiedLeg:
    leg: Leg
    roles: tuple[LegRole, ...]  # a leg can carry multiple roles
    magnitude_atr: float
    crossed_vwap: bool
    reached_hod: bool
    reached_lod: bool
    start_vs_vwap: str  # 'above' | 'below' | 'at'
    end_vs_vwap: str

    @property
    def direction(self) -> str:
        return self.leg.direction


class VwapEventType(str, Enum):
    RECLAIM = "reclaim"  # crossed from below to above
    LOSS = "loss"  # crossed from above to below
    RETEST_HOLD = "retest_hold"  # approached then bounced away (held as S/R)
    RETEST_FAIL = "retest_fail"  # approached then broke through


@dataclass(frozen=True)
class VwapEvent:
    type: VwapEventType
    index: int
    time: pd.Timestamp
    price: float
    vwap: float
    # outcome: signed continuation over the next `horizon` bars in the event's
    # implied direction, in ATR units.
    outcome_atr: float
    horizon_bars: int


@dataclass(frozen=True)
class VwapEventGroup:
    """A run of same-type VWAP events collapsed into one reported event."""

    type: VwapEventType
    count: int
    first_time: pd.Timestamp
    last_time: pd.Timestamp
    representative: VwapEvent  # the strongest (largest |outcome_atr|) in the run

    def label(self) -> str:
        if self.count == 1:
            return (
                f"{self.first_time.strftime('%H:%M')} {self.type.value} "
                f"-> {self.representative.outcome_atr:+.2f} ATR"
            )
        return (
            f"{self.first_time.strftime('%H:%M')}-{self.last_time.strftime('%H:%M')} "
            f"{self.type.value} x{self.count} "
            f"-> {self.representative.outcome_atr:+.2f} ATR (peak)"
        )


def group_vwap_events(events: list[VwapEvent], max_gap_bars: int = 4) -> list[VwapEventGroup]:
    """
    Collapse consecutive same-type VWAP events into groups.

    Four retest_fails 5 minutes apart are one event ("tested VWAP from below 4x,
    failed"), not four. We group runs of the same type whose indices are within
    `max_gap_bars` of each other, keeping the strongest as representative.
    """
    if not events:
        return []
    groups: list[VwapEventGroup] = []
    run = [events[0]]
    for e in events[1:]:
        prev = run[-1]
        if e.type is prev.type and (e.index - prev.index) <= max_gap_bars:
            run.append(e)
        else:
            groups.append(_finalize_group(run))
            run = [e]
    groups.append(_finalize_group(run))
    return groups


def _finalize_group(run: list[VwapEvent]) -> VwapEventGroup:
    rep = max(run, key=lambda e: abs(e.outcome_atr))
    return VwapEventGroup(
        type=run[0].type,
        count=len(run),
        first_time=run[0].time,
        last_time=run[-1].time,
        representative=rep,
    )


class LevelKind(str, Enum):
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    PRIOR_HIGH = "prior_high"
    PRIOR_LOW = "prior_low"
    PRIOR_CLOSE = "prior_close"
    OPEN_RANGE_HIGH = "open_range_high"
    OPEN_RANGE_LOW = "open_range_low"


@dataclass(frozen=True)
class LevelEvent:
    kind: LevelKind
    level: float
    index: int
    time: pd.Timestamp
    price: float
    held: bool  # True if level rejected price (held)
    broke: bool  # True if price broke through and continued
    outcome_atr: float  # signed continuation past the level, ATR units


@dataclass(frozen=True)
class SessionDissection:
    symbol: str
    date: pd.Timestamp
    scale_atr: float  # the major scale used for leg classification
    classified_legs: tuple[ClassifiedLeg, ...]
    vwap_events: tuple[VwapEvent, ...]
    level_events: tuple[LevelEvent, ...]
    summary: dict[str, object]

    def narrate(self) -> str:
        """Human-readable event sequence -- the chart read back in words."""
        lines = [
            f"{self.symbol} {self.date.date()} session dissection (scale {self.scale_atr:.1f} ATR):"
        ]
        for cl in self.classified_legs:
            roles = "+".join(r.value for r in cl.roles)
            lines.append(
                f"  {cl.leg.start_time.strftime('%H:%M')}->"
                f"{cl.leg.end_time.strftime('%H:%M')} "
                f"{cl.direction:>4} {cl.leg.start_price:7.2f}->{cl.leg.end_price:7.2f} "
                f"({cl.magnitude_atr:4.1f} ATR) [{roles}]"
            )
        lines.append(
            f"  VWAP events: {len(self.vwap_events)} "
            f"(reclaims={self.summary.get('n_reclaims', 0)}, "
            f"losses={self.summary.get('n_losses', 0)}, "
            f"holds={self.summary.get('n_retest_holds', 0)}, "
            f"fails={self.summary.get('n_retest_fails', 0)})"
        )
        return "\n".join(lines)


def _vs_vwap(price: float, vwap: float, tol: float) -> str:
    if abs(price - vwap) <= tol:
        return "at"
    return "above" if price > vwap else "below"


def _classify_leg(
    leg: Leg,
    session: Session,
    vwap: np.ndarray,
    atr: float,
    dominant_dir: str,
    hod: float,
    lod: float,
    is_last_third: bool,
    tol: float,
) -> ClassifiedLeg:
    roles: list[LegRole] = []
    mag_atr = leg.magnitude / atr if atr > 0 else 0.0

    sv = vwap[leg.start_index]
    ev = vwap[leg.end_index]
    start_side = _vs_vwap(leg.start_price, sv, tol)
    end_side = _vs_vwap(leg.end_price, ev, tol)
    crossed = (start_side == "below" and end_side == "above") or (
        start_side == "above" and end_side == "below"
    )

    reached_hod = leg.end_price >= hod - tol
    reached_lod = leg.end_price <= lod + tol

    # VWAP crossing roles.
    if start_side == "below" and end_side == "above":
        roles.append(LegRole.VWAP_RECLAIM)
    elif start_side == "above" and end_side == "below":
        roles.append(LegRole.VWAP_LOSS)

    # Flush: large, fast directional move (>=1.5 ATR) to an extreme.
    if mag_atr >= 1.5 and (reached_hod or reached_lod):
        roles.append(LegRole.FLUSH)

    # Extreme tests.
    if reached_hod:
        roles.append(LegRole.HOD_TEST)
    if reached_lod:
        roles.append(LegRole.LOD_TEST)

    # Trend vs retrace relative to dominant direction.
    if leg.direction == dominant_dir and mag_atr >= 1.0:
        roles.append(LegRole.TREND_LEG)
    elif leg.direction != dominant_dir:
        # Any counter-dominant move is a retrace (pullback within the day's bias).
        roles.append(LegRole.RETRACE)
    elif leg.direction == dominant_dir and mag_atr < 1.0:
        # Small with-trend move that didn't qualify as a full trend leg.
        roles.append(LegRole.TREND_LEG)

    # Fade: a give-back leg in the last third of the session, counter to the
    # dominant direction, off an extreme.
    if is_last_third and leg.direction != dominant_dir and mag_atr >= 0.5:
        roles.append(LegRole.FADE)

    if not roles:
        roles.append(LegRole.UNCLASSIFIED)

    return ClassifiedLeg(
        leg=leg,
        roles=tuple(dict.fromkeys(roles)),  # dedupe, preserve order
        magnitude_atr=mag_atr,
        crossed_vwap=crossed,
        reached_hod=reached_hod,
        reached_lod=reached_lod,
        start_vs_vwap=start_side,
        end_vs_vwap=end_side,
    )


def _detect_vwap_events(
    session: Session,
    vwap: np.ndarray,
    atr: float,
    horizon: int = 10,
    proximity_atr: float = 0.25,
) -> list[VwapEvent]:
    """
    Detect VWAP crossings (reclaim/loss) and retests (hold/fail).

    A crossing is a sign change of (close - vwap). A retest is an approach within
    `proximity_atr` of VWAP from one side that then moves away (hold) or crosses
    (fail). Outcome is signed continuation over `horizon` bars in ATR units.
    """
    b = session.bars
    close = b["close"].to_numpy(dtype=float)
    times = b["datetime"].to_numpy()
    n = close.size
    diff = close - vwap
    events: list[VwapEvent] = []

    for i in range(1, n):
        # Crossing.
        if diff[i - 1] <= 0 < diff[i] or diff[i - 1] < 0 <= diff[i]:
            etype = VwapEventType.RECLAIM
        elif diff[i - 1] >= 0 > diff[i] or diff[i - 1] > 0 >= diff[i]:
            etype = VwapEventType.LOSS
        else:
            etype = None

        if etype is not None:
            j = min(n - 1, i + horizon)
            move = close[j] - close[i]
            signed = move if etype is VwapEventType.RECLAIM else -move
            events.append(
                VwapEvent(
                    type=etype,
                    index=i,
                    time=pd.Timestamp(times[i]),
                    price=float(close[i]),
                    vwap=float(vwap[i]),
                    outcome_atr=float(signed / atr) if atr > 0 else 0.0,
                    horizon_bars=j - i,
                )
            )
            continue

        # Retest: close near VWAP without a crossing this bar.
        if atr > 0 and abs(diff[i]) <= proximity_atr * atr:
            j = min(n - 1, i + horizon)
            # Hold if price stays on the same side after the approach; fail if it
            # crosses within the horizon.
            side = np.sign(diff[i]) if diff[i] != 0 else np.sign(diff[i - 1])
            future = diff[i + 1 : j + 1]
            crossed = bool((np.sign(future) != side).any()) if future.size else False
            etype = VwapEventType.RETEST_FAIL if crossed else VwapEventType.RETEST_HOLD
            move = close[j] - close[i]
            signed = move * (1 if side >= 0 else -1)
            events.append(
                VwapEvent(
                    type=etype,
                    index=i,
                    time=pd.Timestamp(times[i]),
                    price=float(close[i]),
                    vwap=float(vwap[i]),
                    outcome_atr=float(signed / atr) if atr > 0 else 0.0,
                    horizon_bars=j - i,
                )
            )

    return events


def _detect_level_events(
    session: Session,
    atr: float,
    horizon: int = 10,
    proximity_atr: float = 0.15,
) -> list[LevelEvent]:
    """
    Detect tests of structural levels (session H/L, prior-day H/L/close, opening
    range) with hold-or-break outcome.
    """
    b = session.bars
    high = b["high"].to_numpy(dtype=float)
    low = b["low"].to_numpy(dtype=float)
    close = b["close"].to_numpy(dtype=float)
    times = b["datetime"].to_numpy()
    n = close.size

    levels: dict[LevelKind, float] = {}
    or_high, or_low = session.opening_range(30)
    levels[LevelKind.OPEN_RANGE_HIGH] = or_high
    levels[LevelKind.OPEN_RANGE_LOW] = or_low
    if session.prior_day is not None:
        levels[LevelKind.PRIOR_HIGH] = session.prior_day.high
        levels[LevelKind.PRIOR_LOW] = session.prior_day.low
        levels[LevelKind.PRIOR_CLOSE] = session.prior_day.close

    tol = proximity_atr * atr if atr > 0 else 0.0
    events: list[LevelEvent] = []

    for kind, lvl in levels.items():
        if not np.isfinite(lvl):
            continue
        for i in range(1, n):
            touched = (low[i] - tol) <= lvl <= (high[i] + tol)
            if not touched:
                continue
            j = min(n - 1, i + horizon)
            # Break = closes beyond the level in the breaking direction by horizon.
            broke_up = close[j] > lvl + tol
            broke_down = close[j] < lvl - tol
            broke = bool(broke_up or broke_down)
            held = not broke
            outcome = (close[j] - lvl) / atr if atr > 0 else 0.0
            events.append(
                LevelEvent(
                    kind=kind,
                    level=float(lvl),
                    index=i,
                    time=pd.Timestamp(times[i]),
                    price=float(close[i]),
                    held=held,
                    broke=broke,
                    outcome_atr=float(outcome),
                )
            )
            break  # first test of each level only (others are continuations)

    return events


def dissect_session(
    session: Session,
    decomposition: MultiScaleDecomposition | None = None,
    scale_atr: float | None = None,
) -> SessionDissection:
    """
    Full dissection of one session.

    Uses the coarsest scale with structure for leg classification (the major
    auction skeleton), and detects VWAP + level events across all bars. Pass an
    explicit `scale_atr` to classify legs at a different granularity.
    """
    decomposition = decomposition or decompose(session)
    if scale_atr is None:
        scale_atr = decomposition.primary_scale(n_bars=len(session))
    if scale_atr is None:
        # No structure at any scale -- return an empty dissection rather than fail.
        return SessionDissection(
            symbol=session.symbol,
            date=session.date,
            scale_atr=float("nan"),
            classified_legs=(),
            vwap_events=(),
            level_events=(),
            summary={"note": "no structural pivots found"},
        )

    # Hierarchical skeleton: coarse major legs split at structural pullbacks.
    major = scale_atr
    refine = max((s for s in decomposition.scales if s < major), default=major)
    legs = build_skeleton(decomposition, major_scale=major, refine_scale=refine)
    if not legs:  # fallback to merged major legs
        legs = merge_insignificant_swings(decomposition.legs_by_scale[scale_atr])
    vwap = session.vwap.to_numpy(dtype=float)
    atr = session.atr_mean
    tol = 0.1 * atr if atr > 0 else 0.0
    hod, lod = session.high, session.low

    # Dominant direction: open->close sign.
    dominant_dir = "up" if session.close_price >= session.open_price else "down"
    n = len(session)
    last_third_start = int(n * 2 / 3)

    classified = tuple(
        _classify_leg(
            leg,
            session,
            vwap,
            atr,
            dominant_dir,
            hod,
            lod,
            is_last_third=(leg.start_index >= last_third_start),
            tol=tol,
        )
        for leg in legs
    )

    vwap_events = tuple(_detect_vwap_events(session, vwap, atr))
    level_events = tuple(_detect_level_events(session, atr))

    summary = {
        "dominant_direction": dominant_dir,
        "n_legs": len(classified),
        "n_reclaims": sum(1 for e in vwap_events if e.type is VwapEventType.RECLAIM),
        "n_losses": sum(1 for e in vwap_events if e.type is VwapEventType.LOSS),
        "n_retest_holds": sum(1 for e in vwap_events if e.type is VwapEventType.RETEST_HOLD),
        "n_retest_fails": sum(1 for e in vwap_events if e.type is VwapEventType.RETEST_FAIL),
        "session_range_atr": (hod - lod) / atr if atr > 0 else float("nan"),
        "vwap_final": session.vwap_final,
    }

    return SessionDissection(
        symbol=session.symbol,
        date=session.date,
        scale_atr=scale_atr,
        classified_legs=classified,
        vwap_events=vwap_events,
        level_events=level_events,
        summary=summary,
    )
