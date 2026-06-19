"""
Outcome labels for the self-learning layer.

Each event's feature vector (features.py, causal/past-only) is joined here to what
actually happened NEXT. Labels look strictly FORWARD from the event bar -- that
is correct and necessary: the label is the future we want to predict. The
features look strictly backward. That split is the entire basis of an honest
learning system.

Two label families are produced per event, for two trading styles:

  * BRACKET (the tradeable label): from the event bar, simulate a trade in a
    given direction with a +target ATR and a -stop ATR and a max holding horizon.
    Outcome is the realized R-multiple (target hit = +reward/risk, stop hit =
    -1R, neither = mark-to-market at horizon). This answers the real question:
    "if I took this signal with this stop and target, did it work?" -- which a
    fixed-horizon label cannot, because it ignores the stop on the way.

  * HORIZON (a robust continuous target): signed forward return over a fixed N
    bars, in ATR units. Useful for ranking signal strength and for the
    multiple-testing layer.

Reversal signals use a wide bracket; scalps use a tight bracket. Same machinery,
different parameters, so the scanner can train both from one event stream.

Intrabar path assumption: with OHLC bars we cannot know whether the high or the
low printed first within a bar. We resolve ties CONSERVATIVELY -- if both target
and stop fall inside the same bar's range, assume the STOP filled first. This
biases measured edge DOWNWARD, which is the safe direction for a system that
must not overstate itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..session.session import Session
from .features import EventFeatures, _causal_atr


@dataclass(frozen=True)
class BracketSpec:
    """A trade bracket: target/stop in ATR, max holding in bars, direction."""

    target_atr: float
    stop_atr: float
    max_bars: int
    name: str = "bracket"

    @property
    def reward_risk(self) -> float:
        return self.target_atr / self.stop_atr if self.stop_atr > 0 else 0.0


# Default brackets for the two styles. Tunable from the scanner/dashboard.
REVERSAL_BRACKET = BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=24, name="reversal")
SCALP_BRACKET = BracketSpec(target_atr=0.75, stop_atr=0.5, max_bars=6, name="scalp")


def brackets_for_timeframe(minutes: int) -> dict[str, BracketSpec]:
    """
    Timeframe-aware bracket presets.

    max_bars and the ML horizon are expressed in BARS, so the same constant means
    different wall-clock time at different resolutions (24 bars = 2h at 5min but
    24m at 1min). This scales the holding windows to a consistent INTENT in
    minutes, so a "reversal" is roughly a 1.5-2h swing and a "scalp" is roughly a
    15-30min move regardless of the bar size the user selected.

    Returns {'reversal': BracketSpec, 'scalp': BracketSpec, 'horizon_bars': int}
    where horizon_bars is the default fixed-horizon (the reversal hold).
    """
    minutes = max(1, int(minutes))
    # Intent in minutes -> bars.
    reversal_hold_min = 120  # ~2 hours
    scalp_hold_min = 25  # ~25 minutes
    rev_bars = max(6, round(reversal_hold_min / minutes))
    scalp_bars = max(3, round(scalp_hold_min / minutes))
    return {
        "reversal": BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=rev_bars, name="reversal"),
        "scalp": BracketSpec(target_atr=0.75, stop_atr=0.5, max_bars=scalp_bars, name="scalp"),
        "horizon_bars": rev_bars,
    }


@dataclass(frozen=True)
class EventLabel:
    """Outcome labels for one event under one trade direction."""

    direction: str  # 'long' | 'short'
    # Bracket outcome
    bracket_name: str
    bracket_r: float  # realized R-multiple (signed)
    target_hit: bool
    stop_hit: bool
    bars_held: int
    # Horizon outcome
    horizon_bars: int
    horizon_return_atr: float  # signed forward return in ATR


def _bracket_outcome(
    session: Session,
    entry_index: int,
    entry_price: float,
    atr: float,
    spec: BracketSpec,
    direction: str,
) -> tuple[float, bool, bool, int]:
    """
    Simulate a bracketed trade forward from entry_index. Returns
    (realized_R, target_hit, stop_hit, bars_held).

    Causal: only uses bars strictly AFTER entry_index (the entry bar's close is
    the fill; outcomes are read from subsequent bars). Conservative tie-break:
    stop-first when both levels are inside one bar.
    """
    b = session.bars
    n = len(b)
    long = direction == "long"
    target = entry_price + spec.target_atr * atr * (1 if long else -1)
    stop = entry_price - spec.stop_atr * atr * (1 if long else -1)
    rr = spec.reward_risk

    last = min(entry_index + spec.max_bars, n - 1)
    for j in range(entry_index + 1, last + 1):
        hi = float(b["high"].iloc[j])
        lo = float(b["low"].iloc[j])
        if long:
            hit_target = hi >= target
            hit_stop = lo <= stop
        else:
            hit_target = lo <= target
            hit_stop = hi >= stop
        if hit_target and hit_stop:
            # Both in one bar -> assume stop first (conservative).
            return (-1.0, False, True, j - entry_index)
        if hit_stop:
            return (-1.0, False, True, j - entry_index)
        if hit_target:
            return (rr, True, False, j - entry_index)

    # Neither hit by horizon: mark to market at the last bar's close, in R units.
    exit_price = float(b["close"].iloc[last])
    move = (exit_price - entry_price) if long else (entry_price - exit_price)
    r = move / (spec.stop_atr * atr) if spec.stop_atr * atr > 0 else 0.0
    return (float(r), False, False, last - entry_index)


def _horizon_return(
    session: Session,
    entry_index: int,
    entry_price: float,
    atr: float,
    horizon: int,
    direction: str,
) -> float:
    """Signed forward return over `horizon` bars in ATR units (causal forward)."""
    b = session.bars
    j = min(entry_index + horizon, len(b) - 1)
    if j <= entry_index:
        return 0.0
    exit_price = float(b["close"].iloc[j])
    move = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    return move / atr if atr > 0 else 0.0


def label_event(
    session: Session,
    event: EventFeatures,
    spec: BracketSpec,
    direction: str,
    horizon: int = 12,
) -> EventLabel:
    """Attach bracket + horizon outcomes to one event for one direction."""
    atr = _causal_atr(session, event.event_index)
    r, tgt, stp, held = _bracket_outcome(
        session, event.event_index, event.event_price, atr, spec, direction
    )
    hz = _horizon_return(session, event.event_index, event.event_price, atr, horizon, direction)
    return EventLabel(
        direction=direction,
        bracket_name=spec.name,
        bracket_r=r,
        target_hit=tgt,
        stop_hit=stp,
        bars_held=held,
        horizon_bars=horizon,
        horizon_return_atr=hz,
    )


def label_events(
    session: Session,
    events: list[EventFeatures],
    spec: BracketSpec = REVERSAL_BRACKET,
    directions: tuple[str, ...] = ("long", "short"),
    horizon: int = 12,
) -> list[dict[str, object]]:
    """
    Build labeled rows: each event x each direction -> features + outcomes.

    Returns flat dicts (feature columns prefixed f_, label columns prefixed y_)
    ready to assemble into a training DataFrame. Both directions are labeled so
    the model can learn long AND short setups (reversals fire both ways).
    """
    rows: list[dict[str, object]] = []
    for ev in events:
        for direction in directions:
            lab = label_event(session, ev, spec, direction, horizon)
            row = ev.as_dict()
            row.update(
                {
                    "y_direction": direction,
                    "y_bracket": spec.name,
                    "y_bracket_r": lab.bracket_r,
                    "y_target_hit": 1 if lab.target_hit else 0,
                    "y_stop_hit": 1 if lab.stop_hit else 0,
                    "y_bars_held": lab.bars_held,
                    "y_horizon_return_atr": lab.horizon_return_atr,
                    "y_win": 1 if lab.bracket_r > 0 else 0,
                }
            )
            rows.append(row)
    return rows
