"""
Causal-boundary re-lock: the pivot-entry lookahead (found via real-data signals).

Two distinct lookaheads were closed here, both invisible in synthetic-only tests
and unmasked only when the NaN fix let the model actually fire on REAL data:

  1. ENTRY TIMING. Events sat at zigzag PIVOT bars (local extremes). A pivot is
     only confirmed once price reverses past it (Pivot.confirmed_index > index),
     so entering/labeling at the extreme is a fill you can't get live. The
     extreme case -- a long from the literal session low -- is a guaranteed win
     (nothing trades below the low), giving `leg_reached_lod -> y_win` AUC = 1.000.
     Fix: leg events now key off leg.decision_index (the confirmation bar) and
     enter at that bar's close.

  2. CONTAMINATED FEATURES. The dissection's leg ROLES use full-session extremes
     (session.high/low) and dominant direction (open vs the session CLOSE) -- both
     future info at the decision bar -- and level_held reads a future bar. These
     are excluded from the ML feature set; causal running-extreme features replace
     them.

These tests assert the leaked features are gone, that the decision bar is the
confirmation bar, and that the leg-feature builder is invariant to anything after
confirmation.
"""

from __future__ import annotations

import _synth as S
import numpy as np

from engine.ml.features import (
    _causal_atr,
    _causal_leg_features,
    extract_session_events,
)
from engine.session.dissect import dissect_session
from engine.session.pivots import build_skeleton, decompose
from engine.session.session import Session

# Features that encoded full-session extremes / dominant direction / a future bar.
BANNED = {
    "f_role_flush",
    "f_role_reversal",
    "f_role_trend",
    "f_role_retrace",
    "f_role_hod_test",
    "f_role_lod_test",
    "f_leg_reached_hod",
    "f_leg_reached_lod",
    "f_level_held",
}


def _events(seed: int):
    ses = S.multileg_session(np.random.default_rng(seed), n_legs=8)
    dis = dissect_session(ses, decomposition=decompose(ses))
    return ses, extract_session_events(ses, dis)


def test_no_lookahead_features_are_emitted():
    """The contaminated feature keys must never appear in any event vector."""
    _ses, events = _events(5)
    assert events
    for ev in events:
        leaked = BANNED & {f"f_{k}" for k in ev.features}
        assert not leaked, f"{ev.event_type} still emits leaked features: {leaked}"


def test_leg_events_decide_at_confirmation_not_the_extreme():
    """At least one leg event's decision bar is its pivot CONFIRMATION (later than
    the extreme), and the entry price is that bar's close -- the realistic fill."""
    ses, events = _events(5)
    legs = [e for e in events if e.event_type == "leg_complete"]
    assert legs
    # confirmation_lag_bars = decision_index - pivot extreme; >0 proves the shift.
    lags = [e.features.get("confirmation_lag_bars", 0.0) for e in legs]
    assert max(lags) > 0.0, "no leg event decided after its extreme -- fix inactive"
    # Entry price equals the close at the decision bar (not the pivot extreme).
    closes = ses.bars["close"].to_numpy(dtype=float)
    for e in legs:
        assert e.event_price == closes[e.event_index]


def test_causal_leg_features_invariant_after_confirmation():
    """The leg feature builder uses only bars up to the confirmation bar: corrupt
    everything after it and the vector is bit-identical (negative control proves
    the corruption is real)."""
    ses = S.multileg_session(np.random.default_rng(3), n_legs=8)
    dec = decompose(ses)
    scale = dec.primary_scale(n_bars=len(ses)) or dec.coarsest_with_pivots()
    legs = build_skeleton(dec, scale)
    leg = next(
        (lg for lg in legs if 0 <= lg.confirmed_index < len(ses) - 1),
        None,
    )
    assert leg is not None, "need a leg confirmed before the final bar"
    idx = leg.decision_index

    a1 = _causal_atr(ses, idx)
    f1 = _causal_leg_features(ses, leg, idx, a1)
    corrupt = Session.from_intraday("TEST", S.corrupt_future(ses.bars, idx))
    a2 = _causal_atr(corrupt, idx)
    f2 = _causal_leg_features(corrupt, leg, idx, a2)

    assert a1 == a2
    assert f1 == f2, "leg features changed when the future was corrupted (leak)"
    # Negative control: the corruption genuinely moved the full-session ATR.
    assert ses.atr_mean != corrupt.atr_mean
