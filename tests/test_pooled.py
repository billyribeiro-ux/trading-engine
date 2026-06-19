"""
Pooled cross-symbol training: concatenation/time-ordering + the power to confirm
or reject an edge with confidence.

Offline via an injected ScannerConfig.frame_builder (synthetic per-symbol frames),
so no network. A persistent edge pooled across symbols is promoted; pooled noise
is not. GBT excluded here only for speed — the gate is identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forward.bakeoff import ModelVersion
from engine.forward.pooled import build_pooled_frame, pooled_bakeoff
from engine.ml.labels import BracketSpec
from engine.ml.signals import ScannerConfig

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
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.arange(N_PER), unit="D")
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


def _config(symbols: list[str], *, persistent: bool) -> ScannerConfig:
    seeds = {s: i + 1 for i, s in enumerate(symbols)}
    return ScannerConfig(
        frame_builder=lambda sym: _sym_frame(sym, seeds[sym], persistent=persistent),
        current_provider=lambda sym: [],
        bracket=BracketSpec(2.0, 1.0, max_bars=2, name="t"),
    )


def test_build_pooled_frame_concats_and_time_sorts():
    syms = ["AAA", "BBB", "CCC"]
    pooled = build_pooled_frame(syms, _config(syms, persistent=True), direction="long")
    assert set(pooled["symbol"]) == set(syms)
    assert len(pooled) == 3 * N_PER
    # globally time-ordered by date
    d = pd.to_datetime(pooled["date"]).to_numpy()
    assert (d[:-1] <= d[1:]).all()


def test_pooled_persistent_edge_is_promoted():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    _, rows = pooled_bakeoff(
        syms,
        _config(syms, persistent=True),
        versions=[ModelVersion("logistic", "logistic")],
        direction="long",
    )
    assert rows and rows[0].promoted, "a persistent edge pooled across symbols must promote"
    assert rows[0].result.realized_edge_r > 0


def test_pooled_noise_promotes_nothing():
    syms = ["AAA", "BBB", "CCC", "DDD"]
    _, rows = pooled_bakeoff(
        syms,
        _config(syms, persistent=False),
        versions=[ModelVersion("logistic", "logistic")],
        direction="long",
    )
    assert not any(r.promoted for r in rows)
