"""
Bracket label correctness + the conservative tie-break (HARD RULE #8).

Labels look FORWARD (that is correct -- the label is the future we predict).
Features look backward. This file plants KNOWN forward paths after an event bar
and asserts the realized R-multiple exactly, including:

  * winner: target hit before stop  -> +reward_risk
  * loser:  stop hit                 -> -1R
  * tie:    one bar spans BOTH       -> -1R (stop-first, the conservative rule)
  * neither: mark-to-market at horizon
  * short side mirrors long

The session is built so causal ATR at the event bar is EXACTLY a known value
(constant-true-range pre-bars), making every target/stop price hand-computable.
"""

from __future__ import annotations

import _synth as S
import pytest

from engine.ml.features import EventFeatures, _causal_atr
from engine.ml.labels import (
    REVERSAL_BRACKET,
    SCALP_BRACKET,
    BracketSpec,
    brackets_for_timeframe,
    label_event,
    label_events,
)

PRE_N = 10  # 10 constant-TR bars -> event bar index 9
BASE = 100.0
TR = 1.0  # causal ATR at the event bar == 1.0
SPEC = BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=5, name="t")  # target 102, stop 99


def _event(session) -> EventFeatures:
    idx = PRE_N - 1
    return EventFeatures(
        symbol="TEST",
        date=session.date,
        event_type="leg_complete",
        event_index=idx,
        event_time=session.bars["datetime"].iloc[idx],
        event_price=BASE,
        features={},
    )


def _session(path_bars):
    return S.append_path_session(PRE_N, path_bars, base=BASE, tr=TR)


def test_atr_precondition_holds():
    """The whole file relies on causal ATR == TR at the event bar."""
    ses = _session([{"open": 100, "high": 100.5, "low": 99.5, "close": 100}])
    assert _causal_atr(ses, PRE_N - 1) == pytest.approx(TR)


def test_long_winner_hits_target():
    # target = 102, stop = 99. Bar reaches 102.5 high, low stays above stop.
    ses = _session([{"open": 100.5, "high": 102.5, "low": 100.2, "close": 102.0}])
    lab = label_event(ses, _event(ses), SPEC, "long")
    assert lab.target_hit and not lab.stop_hit
    assert lab.bracket_r == pytest.approx(SPEC.reward_risk)  # == 2.0
    assert lab.bracket_r == pytest.approx(2.0)


def test_long_loser_hits_stop():
    # low pierces 99, high never reaches 102.
    ses = _session([{"open": 99.8, "high": 99.9, "low": 98.5, "close": 99.0}])
    lab = label_event(ses, _event(ses), SPEC, "long")
    assert lab.stop_hit and not lab.target_hit
    assert lab.bracket_r == pytest.approx(-1.0)


def test_tie_break_is_stop_first():
    """A single bar spanning BOTH target and stop must resolve to the STOP
    (conservative): we cannot know intrabar order, so bias edge downward."""
    ses = _session([{"open": 100.0, "high": 102.5, "low": 98.5, "close": 100.0}])
    lab = label_event(ses, _event(ses), SPEC, "long")
    assert lab.stop_hit and not lab.target_hit
    assert lab.bracket_r == pytest.approx(-1.0)


def test_long_neither_marks_to_market():
    # Drifts up to 100.5 but never touches 102 or 99; exit at horizon close.
    path = [{"open": 100.2, "high": 100.6, "low": 99.8, "close": 100.5}] * 5
    ses = _session(path)
    lab = label_event(ses, _event(ses), SPEC, "long")
    assert not lab.target_hit and not lab.stop_hit
    # r = (exit_close - entry) / (stop_atr * atr) = (100.5 - 100) / (1 * 1) = 0.5
    assert lab.bracket_r == pytest.approx(0.5)


def test_short_winner_hits_target():
    # short: target = 98, stop = 101. Low pierces 98, high stays below stop.
    ses = _session([{"open": 99.5, "high": 99.8, "low": 97.5, "close": 98.0}])
    lab = label_event(ses, _event(ses), SPEC, "short")
    assert lab.target_hit and not lab.stop_hit
    assert lab.bracket_r == pytest.approx(2.0)


def test_short_loser_hits_stop():
    # short stop = 101. high pierces 101, low never reaches 98.
    ses = _session([{"open": 100.5, "high": 101.5, "low": 100.1, "close": 101.0}])
    lab = label_event(ses, _event(ses), SPEC, "short")
    assert lab.stop_hit and not lab.target_hit
    assert lab.bracket_r == pytest.approx(-1.0)


def test_label_events_emits_feature_and_label_columns():
    ses = _session([{"open": 100.5, "high": 102.5, "low": 100.2, "close": 102.0}])
    rows = label_events(ses, [_event(ses)], SPEC, directions=("long", "short"))
    assert len(rows) == 2  # one per direction
    row = rows[0]
    assert "y_bracket_r" in row and "y_win" in row and "y_target_hit" in row
    # y_win mirrors the sign of bracket_r.
    for r in rows:
        assert r["y_win"] == (1 if r["y_bracket_r"] > 0 else 0)


def test_brackets_for_timeframe_scales_with_minutes():
    one = brackets_for_timeframe(1)
    five = brackets_for_timeframe(5)
    # ~2h reversal: 120 bars at 1min, 24 at 5min.
    assert one["reversal"].max_bars == 120
    assert five["reversal"].max_bars == 24
    # ~25m scalp: 25 bars at 1min, 5 at 5min.
    assert one["scalp"].max_bars == 25
    assert five["scalp"].max_bars == 5
    assert five["horizon_bars"] == five["reversal"].max_bars
    # reward/risk geometry is fixed regardless of timeframe.
    assert one["reversal"].reward_risk == pytest.approx(2.0)
    assert one["scalp"].reward_risk == pytest.approx(1.5)
    # Module defaults unchanged.
    assert REVERSAL_BRACKET.reward_risk == pytest.approx(2.0)
    assert SCALP_BRACKET.reward_risk == pytest.approx(1.5)
