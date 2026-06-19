"""
ML feature extraction + the CAUSALITY re-lock (historical bug #1).

Bug #1: features were normalized by full-session ATR, which includes future bars.
A model trained on that leaks the future into the input and dies live. The fix:
`_causal_atr(session, idx)` uses only bars[0:idx+1], and `_session_context`
normalizes by it.

These tests prove causality with TEETH: we corrupt every bar AFTER an index and
assert the causal builders at that index are bit-identical, WHILE demonstrating
that the full-session `atr_mean` DID change under the same corruption. If the
extractor ever reverts to `session.atr_mean`, the negative control guarantees
this test fails.
"""

from __future__ import annotations

import _synth as S
import numpy as np

from engine.ml.features import (
    EVENT_TYPES,
    _causal_atr,
    _session_context,
    extract_session_events,
)
from engine.session.dissect import dissect_session
from engine.session.pivots import decompose
from engine.session.session import Session

CAUSAL_INDICES = (10, 30, 50)


def _dissect(session):
    return dissect_session(session, decomposition=decompose(session))


def test_extraction_produces_causal_event_vectors():
    ses = S.multileg_session(np.random.default_rng(7), n_legs=8)
    events = extract_session_events(ses, _dissect(ses))
    assert events, "expected at least one structural event"
    for ev in events:
        assert ev.event_type in EVENT_TYPES
        assert ev.features, "event must carry a feature vector"
        assert all(np.isfinite(v) for v in ev.features.values())
        # event_index is a valid decision bar within the session
        assert 0 <= ev.event_index < len(ses)
    # Events are sorted by decision bar.
    idxs = [ev.event_index for ev in events]
    assert idxs == sorted(idxs)
    # as_dict prefixes feature keys with f_ (the dataset contract).
    d = events[0].as_dict()
    assert any(k.startswith("f_") for k in d)


def test_causal_atr_invariant_under_future_corruption():
    """_causal_atr(idx) must not move when bars after idx are corrupted."""
    ses = S.multileg_session(np.random.default_rng(3), n_legs=8)
    df = ses.bars
    assert len(ses) > max(CAUSAL_INDICES) + 1
    for idx in CAUSAL_INDICES:
        corrupted = Session.from_intraday("TEST", S.corrupt_future(df, idx))
        a_orig = _causal_atr(ses, idx)
        a_corr = _causal_atr(corrupted, idx)
        assert a_orig == a_corr, f"CAUSAL ATR LEAK at idx {idx}: {a_orig} != {a_corr}"
        # Negative control: the corruption is real -- full-session ATR DID change.
        assert ses.atr_mean != corrupted.atr_mean, (
            "future corruption did not change full-session atr_mean; the test would be vacuous"
        )


def test_session_context_invariant_under_future_corruption():
    """Every causal context feature at idx is bit-identical under future garbage."""
    ses = S.multileg_session(np.random.default_rng(11), n_legs=8)
    df = ses.bars
    for idx in CAUSAL_INDICES:
        corrupted = Session.from_intraday("TEST", S.corrupt_future(df, idx))
        ctx_orig = _session_context(ses, idx)
        ctx_corr = _session_context(corrupted, idx)
        assert ctx_orig.keys() == ctx_corr.keys()
        for k in ctx_orig:
            assert ctx_orig[k] == ctx_corr[k], (
                f"FEATURE LEAK '{k}' at idx {idx}: {ctx_orig[k]} != {ctx_corr[k]}"
            )


def test_causal_atr_differs_from_full_session_atr():
    """Sanity: on a session with non-uniform volatility, causal ATR early in the
    day is NOT the full-session ATR -- proving the two are genuinely different
    quantities (so using the wrong one is a real, detectable leak)."""
    # Calm open, violent afternoon: causal ATR at bar 10 << full-session ATR.
    violent = np.cumsum(np.tile([3.0, -3.0], 20))  # large swings
    closes = 100.0 + np.concatenate([np.cumsum(np.full(20, 0.01)), violent])
    ses = S.session_from_closes(closes, wick=0.02)
    assert _causal_atr(ses, 10) < ses.atr_mean
