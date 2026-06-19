"""
The gauntlet verdict: PASS requires a model to be BOTH promoted (bake-off) and
robust (rolling). A persistent edge passes; noise does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forward.gauntlet import render_gauntlet, run_gauntlet
from engine.ml.labels import BracketSpec
from engine.ml.signals import ScannerConfig

N_PER = 320


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


def _config(symbols, *, persistent: bool) -> ScannerConfig:
    seeds = {s: i + 1 for i, s in enumerate(symbols)}
    return ScannerConfig(
        frame_builder=lambda s: _sym_frame(s, seeds[s], persistent=persistent),
        current_provider=lambda s: [],
        bracket=BracketSpec(2.0, 1.0, max_bars=2, name="t"),
    )


def test_gauntlet_passes_on_persistent_edge():
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    v = run_gauntlet(
        syms,
        _config(syms, persistent=True),
        models=("logistic",),
        direction="long",
        min_holdout_days=5,
        n_windows=4,
    )
    assert v.passed
    assert "PASSED=True" in render_gauntlet(v)


def test_gauntlet_fails_on_noise():
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    v = run_gauntlet(
        syms,
        _config(syms, persistent=False),
        models=("logistic",),
        direction="long",
        min_holdout_days=5,
        n_windows=4,
    )
    assert not v.passed
