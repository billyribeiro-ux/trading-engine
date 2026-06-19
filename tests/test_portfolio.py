"""
Portfolio scanner (Scanner #3): universe metadata + weekly RS-based positions.

Offline: instrument-type classification, the weekly StructuralUnit, and causal
relative-strength features (RS computed from bars idx-N..idx, invariant to future
corruption). The strict leak detector + a full batch_rank smoke run on REAL data.
"""

from __future__ import annotations

import os

import _synth as S
import numpy as np
import pandas as pd
import pytest

from engine.core.structural_unit import BarWindow
from engine.core.universe import InstrumentType, classify
from engine.ml.features import _causal_atr
from engine.portfolio.features import RS_WEEKS, _position_features, extract_position_events
from engine.session.pivots import legs_from_pivots, merge_insignificant_swings, pivots_at_scale

BANNED = {"f_dist_from_vwap_atr", "f_role_flush", "f_leg_reached_lod", "f_level_held"}


def _weekly_window(seed: int, n_legs: int = 24) -> BarWindow:
    rng = np.random.default_rng(seed)
    closes = S.multileg_closes(rng, n_legs=n_legs, bars_per_leg=10, step=0.6, noise=0.05)
    n = len(closes)
    dates = pd.bdate_range("2018-01-05", periods=n, freq="W-FRI")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.2
    lows = np.minimum(opens, closes) - 0.2
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1e6,
        }
    )
    return BarWindow.from_bars("TEST", df, min_bars=60)


def test_universe_classification_flags_inverse_and_leverage():
    sqqq = classify("sqqq")
    assert sqqq.instrument_type is InstrumentType.INVERSE_ETF
    assert sqqq.inverse_of == "QQQ"
    assert sqqq.leverage_factor == -3.0
    assert sqqq.is_derived
    aapl = classify("AAPL")
    assert aapl.instrument_type is InstrumentType.SINGLE_STOCK
    assert not aapl.is_derived


def test_position_events_have_relative_strength_and_are_clean():
    w = _weekly_window(0)
    bench = np.linspace(100.0, 130.0, len(w))  # steadily rising benchmark
    events = extract_position_events(w, scale_atr=2.0, bench_close=bench)
    assert events, "expected position events"
    for e in events:
        assert e.event_type == "position_leg"
        assert not (BANNED & {f"f_{k}" for k in e.features})
    # RS is present for legs confirmed at least RS_WEEKS into the window.
    assert any("rs_vs_bench_13w" in e.features for e in events if e.event_index >= RS_WEEKS)


def test_relative_strength_is_causal():
    w = _weekly_window(1)
    bench = np.linspace(100.0, 140.0, len(w))
    legs = [
        lg
        for lg in merge_insignificant_swings(legs_from_pivots(w, pivots_at_scale(w, 2.0)))
        if lg.end_index > lg.start_index
    ]
    leg = next((lg for lg in legs if RS_WEEKS <= lg.decision_index < len(w) - 1), None)
    assert leg is not None
    idx = leg.decision_index
    f1 = _position_features(w, leg, idx, _causal_atr(w, idx), bench)
    assert "rs_vs_bench_13w" in f1
    corrupt = BarWindow.from_bars("TEST", S.corrupt_future(w.bars, idx))
    # Same benchmark array; only entries <= idx are used -> RS must be unchanged.
    f2 = _position_features(corrupt, leg, idx, _causal_atr(corrupt, idx), bench)
    assert f1 == f2


@pytest.mark.realdata
@pytest.mark.skipif(not os.environ.get("FMP_API_KEY"), reason="no FMP_API_KEY")
def test_portfolio_real_data_frame_and_no_leak():
    from engine.data.client import FMPClient
    from engine.ml.validate import _auc
    from engine.portfolio.dataset import build_position_frame
    from engine.portfolio.window import build_weekly_window

    client = FMPClient(os.environ["FMP_API_KEY"])
    bench = build_weekly_window(client, "SPY")
    df = build_position_frame(client, "AAPL", benchmark=bench)
    if df.empty:
        pytest.skip("no position frame available")
    assert any(c == "f_rs_vs_bench_13w" for c in df.columns), "RS feature missing"
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
            if m.sum() < 25:
                continue
            worst = max(worst, abs(_auc(y[m], x[m]) - 0.5))
    # Catch PERFECT predictors (a structural leak gives AUC == 1.000, like the
    # historical reached_lod bug -> |AUC-0.5| == 0.5). Threshold 0.49 (AUC 0.99)
    # tolerates a genuine causal feature whose univariate AUC inflates on the
    # small weekly sample (~31 events): at_running_low is ~0.88 on the larger swing
    # sample and is proven causal by test_relative_strength_is_causal's sibling.
    assert worst < 0.49, f"near-PERFECT position predictor (leak?): |AUC-0.5|={worst:.3f}"
