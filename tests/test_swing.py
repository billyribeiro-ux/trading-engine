"""
Swing scanner (Scanner #2): the StructuralUnit abstraction + causal swing events.

Proves the three-scanner architecture holds: a generic BarWindow is a
StructuralUnit (as is Session), the SAME pivot decomposition finds swing legs on
daily bars, and the swing features are causal (decided at the pivot confirmation
bar, no intraday/lookahead features). The strict leak detector runs on REAL data
(small synthetic swing samples make univariate AUC unreliable; the brief: verify
on real data).
"""

from __future__ import annotations

import os

import _synth as S
import numpy as np
import pandas as pd
import pytest

from engine.core.structural_unit import BarWindow, StructuralUnit
from engine.ml.features import _causal_atr
from engine.session.pivots import legs_from_pivots, merge_insignificant_swings, pivots_at_scale
from engine.swing.features import _swing_features, extract_swing_events

# Intraday-only / known-leak feature keys that must NEVER appear on a swing event.
BANNED = {"f_dist_from_vwap_atr", "f_role_flush", "f_leg_reached_lod", "f_level_held"}


def _daily_window(seed: int, n_legs: int = 30) -> BarWindow:
    rng = np.random.default_rng(seed)
    closes = S.multileg_closes(rng, n_legs=n_legs, bars_per_leg=12, step=0.5, noise=0.05)
    n = len(closes)
    dates = pd.bdate_range("2023-01-02", periods=n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.15
    lows = np.minimum(opens, closes) - 0.15
    df = pd.DataFrame(
        {"date": dates, "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1e6}
    )
    return BarWindow.from_bars("TEST", df)


def test_barwindow_and_session_are_structural_units():
    assert isinstance(_daily_window(0), StructuralUnit)
    # Session satisfies the same protocol structurally -> one pipeline, two units.
    assert isinstance(S.multileg_session(np.random.default_rng(0)), StructuralUnit)


def test_barwindow_construction_and_atr():
    w = _daily_window(1)
    assert len(w) > 60
    assert w.atr_mean > 0
    with pytest.raises(ValueError):  # missing OHLC columns
        BarWindow.from_bars("X", pd.DataFrame({"date": [1, 2, 3], "open": [1, 2, 3]}))
    with pytest.raises(ValueError):  # too few bars
        BarWindow.from_bars("X", w.bars.head(5), min_bars=60)


def test_swing_events_are_causal_and_clean():
    w = _daily_window(2)
    events = extract_swing_events(w, scale_atr=1.5)
    assert events, "expected swing events"
    closes = w.bars["close"].to_numpy(dtype=float)
    for e in events:
        assert e.event_type == "swing_leg"
        assert e.event_price == closes[e.event_index]  # entry = confirmation close
        assert not (BANNED & {f"f_{k}" for k in e.features}), "intraday/leak feature on a swing"
    # at least one leg decided AFTER its extreme (confirmation), i.e. causal entry.
    assert max(e.features["confirmation_lag_bars"] for e in events) > 0


def test_swing_features_invariant_after_confirmation():
    """Corrupt every bar after a leg's confirmation -> swing features unchanged."""
    w = _daily_window(3)
    legs = [
        lg
        for lg in merge_insignificant_swings(legs_from_pivots(w, pivots_at_scale(w, 1.5)))
        if lg.end_index > lg.start_index
    ]
    leg = next((lg for lg in legs if 0 <= lg.confirmed_index < len(w) - 1), None)
    assert leg is not None
    idx = leg.decision_index
    f1 = _swing_features(w, leg, idx, _causal_atr(w, idx))
    corrupt = BarWindow.from_bars("TEST", S.corrupt_future(w.bars, idx))
    f2 = _swing_features(corrupt, leg, idx, _causal_atr(corrupt, idx))
    assert f1 == f2
    assert w.atr_mean != corrupt.atr_mean  # negative control: corruption is real


@pytest.mark.realdata
@pytest.mark.skipif(not os.environ.get("FMP_API_KEY"), reason="no FMP_API_KEY")
def test_swing_frame_real_data_has_no_perfect_predictor():
    """Leak detector on REAL daily swings: no causal feature may have univariate
    AUC near 0 or 1 vs y_win."""
    from engine.data.client import FMPClient
    from engine.ml.validate import _auc
    from engine.swing.dataset import build_swing_frame

    df = build_swing_frame(FMPClient(os.environ["FMP_API_KEY"]), "AAPL", lookback_days=1460)
    if df.empty:
        pytest.skip("no swing frame available")
    fcols = [c for c in df.columns if c.startswith("f_")]
    worst = 0.0
    for direction in ("long", "short"):
        sub = df[df["y_direction"] == direction]
        y = sub["y_win"].to_numpy(dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for col in fcols:
            x = sub[col].to_numpy(dtype=float)
            m = np.isfinite(x)
            if m.sum() < 30:
                continue
            worst = max(worst, abs(_auc(y[m], x[m]) - 0.5))
    assert worst < 0.45, f"near-perfect swing predictor (leak?): |AUC-0.5|={worst:.3f}"
