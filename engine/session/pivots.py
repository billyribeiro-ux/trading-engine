"""
Multi-scale pivot decomposition.

"Explore every single move, no ceiling" does not mean one threshold -- it means
detecting swings at EVERY scale and keeping the full hierarchy. A single zigzag
threshold forces a false choice: tight catches every wiggle and buries the
structure; loose catches the major legs and misses the small reversal-to-VWAP
moves that are real. The resolution is to run the directional-change algorithm
at a sweep of thresholds (in ATR units) and retain all of them, nested.

On the TSLA session that means: at a large threshold you get the major legs
(open -> flush to 384.7 -> reversal -> grind -> HOD 402 -> fade). At small
thresholds you get the afternoon VWAP retests and micro-rotations inside those
legs. Both are captured. Nothing is discarded by a threshold decision.

The core is the directional-change (DC) / zigzag confirmation algorithm, which is
causal: a pivot is only confirmed once price reverses by the threshold from the
running extreme. We expose both the confirmed pivots (causal, for live use) and
the pivot times (for retrospective dissection). No lookahead in the confirmed
stream: a pivot at bar i is confirmed at bar j>i, and we record both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .session import Session


class PivotType(str, Enum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing pivot at one scale."""

    type: PivotType
    index: int  # bar index of the extreme
    time: pd.Timestamp
    price: float
    confirmed_index: int  # bar index where the reversal confirmed it
    confirmed_time: pd.Timestamp
    scale_atr: float  # the threshold (in ATR units) that detected it


@dataclass(frozen=True)
class Leg:
    """A directional move between two consecutive pivots at one scale."""

    start_index: int
    end_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_price: float
    end_price: float
    scale_atr: float

    @property
    def direction(self) -> str:
        return "up" if self.end_price > self.start_price else "down"

    @property
    def magnitude(self) -> float:
        return abs(self.end_price - self.start_price)

    @property
    def bars(self) -> int:
        return self.end_index - self.start_index


def _directional_change(
    high: np.ndarray,
    low: np.ndarray,
    threshold: float,
) -> list[tuple[PivotType, int, int]]:
    """
    Confirmation-based zigzag (directional change).

    Walks bars, tracking the running extreme since the last confirmed pivot. A
    swing high is confirmed when price falls `threshold` below the running high;
    a swing low when price rises `threshold` above the running low. Returns
    (type, extreme_index, confirmation_index) tuples. Causal: confirmation_index
    is always >= extreme_index.

    threshold is in PRICE units (caller passes scale_atr * atr).
    """
    n = high.size
    if n < 3 or threshold <= 0:
        return []

    pivots: list[tuple[PivotType, int, int]] = []
    # Establish initial direction from the first meaningful move.
    ext_high = high[0]
    ext_high_i = 0
    ext_low = low[0]
    ext_low_i = 0
    direction = 0  # 0 unknown, +1 up-trend (seeking high), -1 down (seeking low)

    for i in range(1, n):
        if high[i] > ext_high:
            ext_high = high[i]
            ext_high_i = i
        if low[i] < ext_low:
            ext_low = low[i]
            ext_low_i = i

        if direction >= 0:
            # Seeking a swing high: confirm if we drop threshold below ext_high.
            if ext_high - low[i] >= threshold:
                pivots.append((PivotType.HIGH, ext_high_i, i))
                direction = -1
                ext_low = low[i]
                ext_low_i = i
                ext_high = high[i]
                ext_high_i = i
                continue
        if direction <= 0:
            # Seeking a swing low: confirm if we rise threshold above ext_low.
            if high[i] - ext_low >= threshold:
                pivots.append((PivotType.LOW, ext_low_i, i))
                direction = 1
                ext_high = high[i]
                ext_high_i = i
                ext_low = low[i]
                ext_low_i = i
                continue

    return pivots


def pivots_at_scale(session: Session, scale_atr: float) -> list[Pivot]:
    """Confirmed pivots at one scale (threshold = scale_atr * session ATR).

    Appends a TERMINAL pivot at the session's final bar so the last (still
    unconfirmed) swing is closed. Without this, a coarse scale leaves the closing
    move untracked -- e.g. a strong end-of-day rip never reverses by the
    threshold to confirm, so the skeleton ends mid-afternoon. The terminal pivot
    is marked confirmed at the final bar and lets coarse scales express the full
    session while finer scales handle the internal detail.
    """
    b = session.bars
    high = b["high"].to_numpy(dtype=float)
    low = b["low"].to_numpy(dtype=float)
    times = b["datetime"].to_numpy()
    atr = session.atr_mean
    if not np.isfinite(atr) or atr <= 0:
        return []
    threshold = scale_atr * atr

    raw = _directional_change(high, low, threshold)
    pivots: list[Pivot] = []
    for ptype, ext_i, conf_i in raw:
        price = high[ext_i] if ptype is PivotType.HIGH else low[ext_i]
        pivots.append(
            Pivot(
                type=ptype,
                index=ext_i,
                time=pd.Timestamp(times[ext_i]),
                price=float(price),
                confirmed_index=conf_i,
                confirmed_time=pd.Timestamp(times[conf_i]),
                scale_atr=scale_atr,
            )
        )

    # Terminal pivot: close the final swing at the last bar so coarse scales
    # cover the whole session. Only add if the last confirmed pivot isn't already
    # at/near the final bar.
    n = len(b)
    last_idx = n - 1
    if pivots and pivots[-1].index < last_idx - 1:
        last_pivot = pivots[-1]
        # The terminal extreme is the high or low since the last pivot, opposite
        # in type to the last pivot (the swing that was in progress).
        seg_hi_i = int(last_pivot.index + 1 + np.argmax(high[last_pivot.index + 1 :]))
        seg_lo_i = int(last_pivot.index + 1 + np.argmin(low[last_pivot.index + 1 :]))
        if last_pivot.type is PivotType.LOW:
            # In-progress swing is up -> terminal is the highest high since.
            t_type, t_i = PivotType.HIGH, seg_hi_i
            t_price = float(high[t_i])
        else:
            t_type, t_i = PivotType.LOW, seg_lo_i
            t_price = float(low[t_i])
        pivots.append(
            Pivot(
                type=t_type,
                index=t_i,
                time=pd.Timestamp(times[t_i]),
                price=t_price,
                confirmed_index=last_idx,
                confirmed_time=pd.Timestamp(times[last_idx]),
                scale_atr=scale_atr,
            )
        )
    elif not pivots:
        # No pivots at all (very coarse scale): create a single leg open->extreme
        # so the session still has a skeleton. Use the larger of the up/down move.
        pass
    return pivots


def legs_from_pivots(session: Session, pivots: list[Pivot]) -> list[Leg]:
    """Build the directional legs connecting consecutive pivots."""
    if len(pivots) < 2:
        return []
    legs: list[Leg] = []
    for a, c in zip(pivots[:-1], pivots[1:]):
        legs.append(
            Leg(
                start_index=a.index,
                end_index=c.index,
                start_time=a.time,
                end_time=c.time,
                start_price=a.price,
                end_price=c.price,
                scale_atr=a.scale_atr,
            )
        )
    return legs


@dataclass(frozen=True)
class MultiScaleDecomposition:
    """The full pivot/leg hierarchy across all scales for one session."""

    session_date: pd.Timestamp
    symbol: str
    scales: tuple[float, ...]
    pivots_by_scale: dict[float, list[Pivot]]
    legs_by_scale: dict[float, list[Leg]]
    atr_mean: float = 1.0

    def all_pivots(self) -> list[Pivot]:
        out: list[Pivot] = []
        for s in self.scales:
            out.extend(self.pivots_by_scale.get(s, []))
        return out

    def coarsest_with_pivots(self) -> float | None:
        """Largest scale that still found >=2 pivots (the major structure)."""
        for s in sorted(self.scales, reverse=True):
            if len(self.pivots_by_scale.get(s, [])) >= 2:
                return s
        return None

    def primary_scale(
        self,
        n_bars: int | None = None,
        target_legs: int = 8,
        min_legs: int = 4,
        max_legs: int = 14,
        end_coverage_frac: float = 0.85,
    ) -> float | None:
        """
        Pick the scale that best captures the MAJOR auction structure.

        Strategy: among scales that cover the session end, evaluate each by its
        MERGED leg count (since the merge now correctly preserves structural
        pullbacks and only removes noise). Prefer the FINEST scale whose merged
        skeleton lands in [min_legs, max_legs] -- finer scales resolve structural
        pullbacks (like a 0.94-ATR midday pullback) that coarse scales miss
        entirely, while the merge keeps the count sane. Tie-break toward the leg
        count closest to target_legs.

        This replaces the prior "coarsest acceptable" rule, which picked 2.0 ATR
        and collapsed a real midday pullback that only exists at 1.0 ATR.
        """
        from .pivots import merge_insignificant_swings  # local to avoid cycle

        candidates: list[tuple[float, int]] = []
        for s in sorted(self.scales):  # ascending: finest first
            raw = self.legs_by_scale.get(s, [])
            if len(raw) < min_legs:
                continue
            if n_bars is not None and raw:
                last_end = max(lg.end_index for lg in raw)
                if last_end < end_coverage_frac * (n_bars - 1):
                    continue
            merged = merge_insignificant_swings(raw)
            if min_legs <= len(merged) <= max_legs:
                candidates.append((s, len(merged)))
        if not candidates:
            return self.coarsest_with_pivots()
        # Finest scale first (already sorted ascending); among those, the one
        # whose merged count is closest to target wins, tie-break to finer.
        candidates.sort(key=lambda x: (abs(x[1] - target_legs), x[0]))
        return candidates[0][0]


def merge_insignificant_swings(
    legs: list[Leg],
    counter_frac_max: float = 0.40,
    counter_bars_max: int = 3,
) -> list[Leg]:
    """
    Absorb only TRULY INSIGNIFICANT counter-swings into the dominant move.

    A counter-swing (b) between two same-direction legs (a, c) is absorbed only
    if ALL of:
      * the move continues to a NEW extreme (c beyond a's end), AND
      * the counter-swing is a small fraction of the combined move
        (magnitude/combined < counter_frac_max), AND
      * it is brief (<= counter_bars_max bars).

    The fraction gate is the decisive one and is checked FIRST: a counter-swing
    that retraces a large share of the move is structural no matter how brief.
    Evidence: the real 11:45 TSLA pullback retraced 78% of its surrounding move
    -- absorbing that (just because it spanned 4 bars) hid a real structural leg.
    A pullback only qualifies as noise when it is BOTH shallow AND brief.
    """
    if len(legs) < 3:
        return legs

    changed = True
    work = list(legs)
    guard = 0
    while changed and guard < 50:
        guard += 1
        changed = False
        out: list[Leg] = []
        i = 0
        while i < len(work):
            if i + 2 <= len(work) - 1:
                a, b, c = work[i], work[i + 1], work[i + 2]
                if a.direction == c.direction and a.direction != b.direction:
                    a_down = a.direction == "down"
                    new_extreme = c.end_price < a.end_price if a_down else c.end_price > a.end_price
                    combined = abs(c.end_price - a.start_price)
                    counter_frac = b.magnitude / combined if combined > 0 else 1.0
                    counter_shallow = counter_frac < counter_frac_max
                    counter_brief = b.bars <= counter_bars_max
                    # Decisive: shallow AND brief AND continues to a new extreme.
                    if new_extreme and counter_shallow and counter_brief:
                        out.append(
                            Leg(
                                start_index=a.start_index,
                                end_index=c.end_index,
                                start_time=a.start_time,
                                end_time=c.end_time,
                                start_price=a.start_price,
                                end_price=c.end_price,
                                scale_atr=a.scale_atr,
                            )
                        )
                        i += 3
                        changed = True
                        continue
            out.append(work[i])
            i += 1
        work = out
    return work


def split_leg_at_structural_pullbacks(
    major: Leg,
    finer_legs: list[Leg],
    min_retrace_frac: float = 0.35,
    min_pullback_frac_of_leg: float = 0.20,
    atr: float = 1.0,
) -> list[Leg]:
    """
    Split one major leg at any STRUCTURAL pullback hidden inside it.

    The major skeleton (coarse scale) is clean but can hide a real counter-trend
    pullback -- e.g. the TSLA reversal 384.81->397.09 hid a pullback to 387.53
    that the coarse scale never surfaced. This finds such pullbacks using the
    finer-scale swings within the major leg and splits it into
    [advance, pullback, advance, ...].

    A counter-swing inside the major leg is STRUCTURAL only if BOTH:
      * it retraces >= min_retrace_frac of the move progressed so far (a real
        giveback, not a pause), AND
      * its magnitude is >= min_pullback_frac_of_leg of the MAJOR LEG's total
        magnitude (significant relative to the leg it would split, not an
        absolute floor).

    The leg-relative second gate is what makes this correct everywhere: it
    protects a clean flush (whose internal bounces are small vs the whole flush)
    AND prevents the afternoon pullback from shattering into micro-swings
    (each tiny swing is a small fraction of its parent leg). Only a giveback
    that is large both in retrace terms AND relative to the parent leg splits it.
    """
    if not finer_legs or major.bars < 2:
        return [major]

    up = major.direction == "up"
    major_mag = major.magnitude
    if major_mag <= 0:
        return [major]

    counters = [
        lg
        for lg in finer_legs
        if lg.direction != major.direction
        and lg.start_index >= major.start_index
        and lg.end_index <= major.end_index
    ]
    if not counters:
        return [major]

    split_points: list[Leg] = []
    for cl in counters:
        if up:
            progressed = cl.start_price - major.start_price
        else:
            progressed = major.start_price - cl.start_price
        if progressed <= 0:
            continue
        retrace = cl.magnitude / progressed
        frac_of_leg = cl.magnitude / major_mag
        if retrace >= min_retrace_frac and frac_of_leg >= min_pullback_frac_of_leg:
            split_points.append(cl)

    if not split_points:
        return [major]

    # Keep non-overlapping pullbacks, earliest-first.
    split_points.sort(key=lambda x: x.start_index)
    chosen: list[Leg] = []
    last_end = major.start_index
    for cl in split_points:
        if cl.start_index >= last_end:
            chosen.append(cl)
            last_end = cl.end_index
    if not chosen:
        return [major]

    pieces: list[Leg] = []
    cursor_idx, cursor_time, cursor_price = (major.start_index, major.start_time, major.start_price)
    for cl in chosen:
        if cl.start_index > cursor_idx:
            pieces.append(
                Leg(
                    start_index=cursor_idx,
                    end_index=cl.start_index,
                    start_time=cursor_time,
                    end_time=cl.start_time,
                    start_price=cursor_price,
                    end_price=cl.start_price,
                    scale_atr=major.scale_atr,
                )
            )
        pieces.append(
            Leg(
                start_index=cl.start_index,
                end_index=cl.end_index,
                start_time=cl.start_time,
                end_time=cl.end_time,
                start_price=cl.start_price,
                end_price=cl.end_price,
                scale_atr=major.scale_atr,
            )
        )
        cursor_idx, cursor_time, cursor_price = (cl.end_index, cl.end_time, cl.end_price)
    if cursor_idx < major.end_index:
        pieces.append(
            Leg(
                start_index=cursor_idx,
                end_index=major.end_index,
                start_time=cursor_time,
                end_time=major.end_time,
                start_price=cursor_price,
                end_price=major.end_price,
                scale_atr=major.scale_atr,
            )
        )
    return pieces if pieces else [major]


def build_skeleton(
    decomposition: MultiScaleDecomposition,
    major_scale: float,
    refine_scale: float | None = None,
    n_bars: int | None = None,
) -> list[Leg]:
    """
    The clean session skeleton: the merged major legs at the primary scale.

    Evidence (real TSLA 2026-06-18): at the primary scale, the merge already
    yields the correct structure -- flush / reversal / midday pullback / grind /
    afternoon pullback / rip. An earlier version then ran a finer-scale "split at
    structural pullbacks" pass on top; on real data that RE-FRAGMENTED the
    already-clean legs (17 legs instead of 6). The merge at the right scale is
    sufficient and correct; the split pass was removed. `refine_scale`/`n_bars`
    are accepted for signature stability but no longer drive splitting.

    Zero-duration legs (a terminal pivot landing on the final bar) are dropped so
    the skeleton has no 0-minute entries.
    """
    major_legs = merge_insignificant_swings(decomposition.legs_by_scale.get(major_scale, []))
    # Drop zero-duration legs (e.g. terminal blip on the last bar).
    skeleton = [lg for lg in major_legs if lg.end_index > lg.start_index]
    return skeleton


def decompose(
    session: Session,
    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0),
) -> MultiScaleDecomposition:
    """
    Multi-scale decomposition of a session.

    `scales` are threshold multiples of the session ATR, from fine (0.25 ATR --
    every micro-swing) to coarse (5 ATR -- only the major legs). The default
    sweep spans micro to major; pass a denser sweep to "explore everything" at
    even finer granularity. There is deliberately no single chosen scale -- the
    hierarchy is the output.
    """
    pivots_by_scale: dict[float, list[Pivot]] = {}
    legs_by_scale: dict[float, list[Leg]] = {}
    for s in scales:
        pv = pivots_at_scale(session, s)
        pivots_by_scale[s] = pv
        legs_by_scale[s] = legs_from_pivots(session, pv)
    return MultiScaleDecomposition(
        session_date=session.date,
        symbol=session.symbol,
        scales=tuple(scales),
        pivots_by_scale=pivots_by_scale,
        legs_by_scale=legs_by_scale,
        atr_mean=session.atr_mean,
    )
