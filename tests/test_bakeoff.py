"""
Version bake-off: only versions with hard FORWARD evidence get promoted.

Proves the evidence gate: a genuinely persistent version is promoted; versions
built on noise features, and ANY version on a decayed edge, are not — even the
more expressive GBT. This is the "100% evidence-gated" discipline, the opposite of
chasing in-sample accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forward.bakeoff import ModelVersion, bake_off, render_bakeoff

FEATS = ["f_signal", "f_n1", "f_n2"]
N = 900


def _frame(seed: int, *, train_signal: bool, holdout_signal: bool, cut_frac: float = 0.7):
    rng = np.random.default_rng(seed)
    cut = int(N * cut_frac)
    f_signal = np.empty(N)
    y_win = np.empty(N, dtype=int)
    for i in range(N):
        on = train_signal if i < cut else holdout_signal
        latent = rng.normal()
        if on:
            y_win[i] = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.5 * latent)))
            f_signal[i] = latent + rng.normal(0.0, 0.5)
        else:
            y_win[i] = rng.integers(0, 2)
            f_signal[i] = rng.normal()
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": pd.Timestamp("2026-01-01"),
            "event_index": np.arange(N),
            "f_signal": f_signal,
            "f_n1": rng.normal(size=N),
            "f_n2": rng.normal(size=N),
            "y_win": y_win,
            "y_bracket_r": np.where(y_win == 1, 2.0, -1.0).astype(float),
        }
    )


def test_persistent_version_promoted_noise_features_not():
    df = _frame(1, train_signal=True, holdout_signal=True)
    versions = [
        ModelVersion("logit_all", "logistic"),
        ModelVersion("logit_noise", "logistic", features=("f_n1", "f_n2")),
        ModelVersion("gbt_all", "gbt"),
    ]
    rows = bake_off(df, FEATS, versions, horizon_bars=2)
    by = {r.version: r for r in rows}
    assert by["logit_all"].promoted, "a persistent edge must be promoted"
    assert not by["logit_noise"].promoted, "noise-only features must not be promoted"
    assert rows[0].promoted  # promoted sorted first
    # every version got an FDR-adjusted forward p in [0, 1]
    assert all(0.0 <= r.p_value_fdr <= 1.0 for r in rows)


def test_decayed_edge_promotes_nothing_even_with_gbt():
    df = _frame(3, train_signal=True, holdout_signal=False)
    rows = bake_off(
        df,
        FEATS,
        [ModelVersion("logistic", "logistic"), ModelVersion("gbt", "gbt")],
        horizon_bars=2,
    )
    assert not any(r.promoted for r in rows)
    assert "no version persists" in render_bakeoff(rows)


def test_pure_noise_promotes_nothing():
    df = _frame(2, train_signal=False, holdout_signal=False)
    rows = bake_off(df, FEATS, horizon_bars=2)  # default slate (logistic + gbt)
    assert not any(r.promoted for r in rows)
