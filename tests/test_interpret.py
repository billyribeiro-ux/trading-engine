"""
Forward feature importance: the driver of the edge is the feature whose
destruction on the holdout kills the realized edge — noise features don't matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.forward.interpret import permutation_importance, render_importance

FEATS = ["f_signal", "f_n1", "f_n2"]
N = 900


def _frame(seed: int):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=N)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.5 * latent)))
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(N), unit="D"),
            "event_index": np.arange(N),
            "f_signal": latent + rng.normal(0.0, 0.5, size=N),  # the real driver
            "f_n1": rng.normal(size=N),
            "f_n2": rng.normal(size=N),
            "y_win": y.astype(int),
            "y_bracket_r": np.where(y == 1, 2.0, -1.0).astype(float),
        }
    )


def test_real_driver_ranks_first_noise_does_not_matter():
    base_edge, base_auc, rows = permutation_importance(
        _frame(1), FEATS, model_kind="logistic", horizon_bars=2, n_repeats=5
    )
    assert base_auc > 0.55
    assert rows[0].feature == "f_signal"
    assert rows[0].auc_drop > 0.05  # destroying the driver hurts forward AUC
    noise = {r.feature: r for r in rows if r.feature != "f_signal"}
    assert all(abs(noise[f].auc_drop) < rows[0].auc_drop for f in noise)
    assert "feature" in render_importance(base_edge, base_auc, rows)
