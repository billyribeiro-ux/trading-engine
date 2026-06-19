"""
Signal generator: the adversarial gate (HARD RULE #5).

A signal may exist ONLY behind validated, FDR-corrected, edge-over-baseline,
net-of-cost OOS performance. These tests prove:

  * NOISE in -> ZERO signals out. A 2:1 bracket is +EV by geometry, so a broken
    generator would happily emit; this asserts it does not.
  * PLANTED edge in -> signals out, with entry/stop/target EXACTLY equal to the
    hand-computed bracket x causal-ATR geometry, and the validation backing
    attached.

Fully offline: the scanner-agnostic seam (frame_builder + current_provider) is
injected with synthetic functions, so no network and full determinism. That same
seam is what lets one code path serve the intraday, swing, and portfolio scanners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.ml.labels import BracketSpec
from engine.ml.signals import (
    ScannerConfig,
    ScorableEvent,
    batch_rank,
    bracket_levels,
    render_signals,
)

N = 800
BRACKET = BracketSpec(target_atr=2.0, stop_atr=1.0, max_bars=10, name="test")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _direction_block(rng, direction, *, planted):
    """N event rows for one direction; planted=True puts real edge in f_signal."""
    latent = rng.normal(size=N)
    if planted:
        y_win = rng.binomial(1, _sigmoid(2.5 * latent))
        f_signal = latent + rng.normal(0.0, 0.5, size=N)
    else:
        y_win = rng.integers(0, 2, size=N)
        f_signal = rng.normal(size=N)
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.Timestamp("2026-06-01"),
            "event_index": np.arange(N),
            "y_direction": direction,
            "f_signal": f_signal,
            "f_n1": rng.normal(size=N),
            "f_n2": rng.normal(size=N),
            "y_win": y_win.astype(int),
            "y_bracket_r": np.where(y_win == 1, 2.0, -1.0).astype(float),  # 2:1 geometry
        }
    )


def _frame_builder(*, long_planted, short_planted):
    def build(_symbol: str) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.concat(
            [
                _direction_block(rng, "long", planted=long_planted),
                _direction_block(rng, "short", planted=short_planted),
            ],
            ignore_index=True,
        )

    return build


def _current_provider(events):
    return lambda _symbol: list(events)


def _config(*, long_planted, short_planted, events):
    return ScannerConfig(
        frame_builder=_frame_builder(long_planted=long_planted, short_planted=short_planted),
        current_provider=_current_provider(events),
        bracket=BRACKET,
        proba_threshold=0.55,
        n_folds=5,
        fdr=0.10,
        min_events=200,
    )


# --- pure geometry ---------------------------------------------------------


def test_bracket_levels_long_and_short_are_hand_computed():
    # entry 100, atr 2, bracket 2:1 -> long stop 98 / target 104; short mirrors.
    assert bracket_levels(100.0, 2.0, BRACKET, "long") == (98.0, 104.0)
    assert bracket_levels(100.0, 2.0, BRACKET, "short") == (102.0, 96.0)
    with pytest.raises(ValueError):
        bracket_levels(100.0, 2.0, BRACKET, "sideways")


# --- the adversarial gate --------------------------------------------------


def test_noise_emits_zero_signals():
    """Pure noise (both directions) must survive nothing -> emit nothing, even
    though the bracket is +EV by geometry."""
    ev = ScorableEvent(
        "vwap_reclaim", pd.Timestamp("2026-06-18 10:00"), 100.0, 2.0, {"signal": 3.0}
    )
    result = batch_rank(["AAA"], _config(long_planted=False, short_planted=False, events=[ev]))
    assert result.signals == ()
    assert result.survivors == ()
    assert len(result.reports) == 2  # long + short both evaluated and rejected
    assert "no validated signals" in render_signals(result)


def test_planted_edge_emits_signal_with_exact_geometry():
    """Edge planted in the LONG direction only -> a long signal whose
    entry/stop/target match the bracket x causal-ATR math exactly."""
    ev = ScorableEvent(
        event_type="vwap_reclaim",
        timestamp=pd.Timestamp("2026-06-18 10:00"),
        price=100.0,
        atr=2.0,
        features={"signal": 3.0},  # high -> long model fires; n1/n2 absent -> imputed
    )
    result = batch_rank(["AAA"], _config(long_planted=True, short_planted=False, events=[ev]))

    assert result.signals, "planted edge produced no signals"
    longs = [s for s in result.signals if s.direction == "long"]
    assert longs, "expected a long signal from the planted-long config"
    s = longs[0]
    # Geometry: entry 100, atr 2, 2:1 bracket -> stop 98, target 104.
    assert s.entry == pytest.approx(100.0)
    assert s.stop == pytest.approx(98.0)
    assert s.target == pytest.approx(104.0)
    assert s.rr == pytest.approx(2.0)
    assert s.event_type == "vwap_reclaim"
    # Validation backing is attached and real.
    assert s.probability >= 0.55
    assert s.p_value_fdr < 0.10
    assert s.oos_edge_r > 0.0
    assert s.oos_auc > 0.60
    # No SHORT signal: the short config was noise and must not survive.
    assert all(sig.direction != "short" for sig in result.signals)


def test_only_surviving_direction_calls_current_provider():
    """If nothing survives, the (potentially expensive) current provider for live
    scoring must not run at all."""
    calls = {"n": 0}

    def counting_provider(_symbol):
        calls["n"] += 1
        return []

    cfg = ScannerConfig(
        frame_builder=_frame_builder(long_planted=False, short_planted=False),
        current_provider=counting_provider,
        bracket=BRACKET,
        min_events=200,
    )
    result = batch_rank(["AAA"], cfg)
    assert result.signals == ()
    assert calls["n"] == 0


def test_low_probability_event_is_not_emitted_even_when_config_survives():
    """A validated config still emits nothing for an event the model rates below
    threshold (a strongly-bearish reading for a long model)."""
    bearish = ScorableEvent(
        "vwap_loss", pd.Timestamp("2026-06-18 11:00"), 100.0, 2.0, {"signal": -4.0}
    )
    result = batch_rank(["AAA"], _config(long_planted=True, short_planted=False, events=[bearish]))
    # The long config survives validation, but this event scores low -> no signal.
    assert result.survivors, "expected the planted-long config to survive"
    assert all(s.probability >= 0.55 for s in result.signals)
    assert not any(s.event_type == "vwap_loss" for s in result.signals)
