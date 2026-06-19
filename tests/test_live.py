"""
Live signal generation — emit signals only when the edge is currently alive.

Offline via an injected ScannerConfig (synthetic pooled frames + a current event):
a persistent edge fires signals with correct bracket geometry; a config with no
forward edge fires NOTHING (the live gate is the same forward-test verdict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forward.live import generate_live_signals
from engine.ml.labels import BracketSpec
from engine.ml.signals import ScannerConfig, ScorableEvent

N_PER = 300


def _sym_frame(symbol: str, seed: int, *, persistent: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=N_PER)
    if persistent:
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.5 * latent)))
        fsig = latent + rng.normal(0.0, 0.5, size=N_PER)
    else:
        y = rng.integers(0, 2, size=N_PER)
        fsig = rng.normal(size=N_PER)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(N_PER), unit="D")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "event_index": np.arange(N_PER),
            "y_direction": "long",
            "f_signal": fsig,
            "f_n1": rng.normal(size=N_PER),
            "f_n2": rng.normal(size=N_PER),
            "y_win": y.astype(int),
            "y_bracket_r": np.where(y == 1, 2.0, -1.0).astype(float),
        }
    )


def _config(symbols, *, persistent: bool, signal_value: float) -> ScannerConfig:
    seeds = {s: i + 1 for i, s in enumerate(symbols)}
    ev = ScorableEvent(
        "swing_leg", pd.Timestamp("2026-06-19"), 100.0, 2.0, {"signal": signal_value}
    )
    return ScannerConfig(
        frame_builder=lambda s: _sym_frame(s, seeds[s], persistent=persistent),
        current_provider=lambda s: [ev],
        bracket=BracketSpec(2.0, 1.0, max_bars=8, name="swing"),
    )


def test_live_emits_with_correct_geometry_when_edge_persists():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    sigs, bt = generate_live_signals(
        syms,
        _config(syms, persistent=True, signal_value=3.0),
        model_kind="logistic",
        direction="long",
        min_holdout_days=5,
    )
    assert bt.persisted
    assert sigs, "a live, persistent edge must emit signals"
    s = sigs[0]
    assert s.direction == "long"
    assert (s.entry, s.stop, s.target) == (100.0, 98.0, 104.0)  # entry 100, atr 2, 2:1
    assert s.max_bars == 8
    assert s.oos_edge_r > 0  # backing = current forward edge


def test_live_emits_nothing_when_no_forward_edge():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    sigs, bt = generate_live_signals(
        syms,
        _config(syms, persistent=False, signal_value=0.0),
        model_kind="logistic",
        direction="long",
        min_holdout_days=5,
    )
    assert sigs == []
    assert not (bt is not None and bt.persisted)
