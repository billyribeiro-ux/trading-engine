"""
Walk-forward validation: the bracket-geometry false-positive re-lock (bug #2).

Bug #2: the harness reported "+0.43R edge, p=0.000" on PURE NOISE, because a 2:1
reward/risk bracket is positive-expectancy BY CONSTRUCTION, with zero skill.
Measuring raw bracket expectancy credits the geometry as if it were edge.

The fix (validate.py): edge is measured as the model's selected-signal mean R
OVER THE BASELINE of taking every event, and the p-value tests that difference.

These tests feed the harness:
  * NOISE: random features, random wins, but R drawn from the SAME profitable 2:1
    bracket -> raw expectancy is positive, yet edge-over-baseline must be ~0 and
    p must NOT be significant (p > 0.10). This is the exact false-positive shape.
  * PLANTED EDGE: a feature genuinely predictive of the outcome -> the model's
    selection beats baseline -> p < 0.05, AUC > 0.6, positive edge.
  * FDR: across [noise, planted], only the planted config survives.

Everything is deterministic: data generation is seeded, the model is closed-form
gradient descent, and the bootstrap p-value is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.ml.validate import fdr_correct_reports, walk_forward_validate

FEATURE_COLS = ["f_signal", "f_n1", "f_n2"]
N = 800


def _frame(seed: int, signal_strength: float) -> pd.DataFrame:
    """Labeled event frame. signal_strength==0 -> pure noise; >0 -> planted edge.

    y_bracket_r is ALWAYS drawn from a profitable 2:1 bracket (+2R win / -1R loss)
    so the baseline is positive-by-geometry in BOTH cases -- that is what makes
    the noise case the genuine false-positive trap.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=N)
    if signal_strength > 0:
        prob = 1.0 / (1.0 + np.exp(-signal_strength * latent))
        y_win = rng.binomial(1, prob)
        f_signal = latent + rng.normal(0.0, 0.5, size=N)  # observable, predictive
    else:
        y_win = rng.integers(0, 2, size=N)
        f_signal = rng.normal(size=N)  # observable, useless
    y_bracket_r = np.where(y_win == 1, 2.0, -1.0)  # 2:1 geometry
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": pd.Timestamp("2026-06-01"),
            "event_index": np.arange(N),
            "f_signal": f_signal,
            "f_n1": rng.normal(size=N),
            "f_n2": rng.normal(size=N),
            "y_win": y_win.astype(int),
            "y_bracket_r": y_bracket_r.astype(float),
        }
    )


def test_noise_is_rejected_despite_profitable_bracket():
    df = _frame(seed=1, signal_strength=0.0)
    # Sanity: the bracket really is positive-expectancy by geometry.
    assert df["y_bracket_r"].mean() > 0.3, "noise baseline should be +EV by geometry"
    rep = walk_forward_validate(df, FEATURE_COLS, n_folds=5, seed=0)
    assert rep.p_value > 0.10, f"noise judged significant (p={rep.p_value})"
    # Edge OVER BASELINE (not raw expectancy) must be ~0.
    assert abs(rep.oos_net_expectancy_r) < 0.25, rep.oos_net_expectancy_r
    # No real ranking ability.
    assert 0.40 <= rep.oos_auc <= 0.60, rep.oos_auc


def test_planted_edge_is_detected():
    df = _frame(seed=2, signal_strength=2.5)
    rep = walk_forward_validate(df, FEATURE_COLS, n_folds=5, seed=0)
    assert rep.p_value < 0.05, f"planted edge missed (p={rep.p_value})"
    assert rep.oos_auc > 0.60, rep.oos_auc
    assert rep.oos_net_expectancy_r > 0.0, rep.oos_net_expectancy_r


def test_validation_is_deterministic():
    df = _frame(seed=2, signal_strength=2.5)
    a = walk_forward_validate(df, FEATURE_COLS, n_folds=5, seed=0)
    b = walk_forward_validate(df, FEATURE_COLS, n_folds=5, seed=0)
    assert a.p_value == b.p_value
    assert a.oos_auc == b.oos_auc
    assert a.oos_net_expectancy_r == b.oos_net_expectancy_r


def test_fdr_keeps_only_the_real_edge():
    noise = walk_forward_validate(_frame(1, 0.0), FEATURE_COLS, n_folds=5, seed=0)
    planted = walk_forward_validate(_frame(2, 2.5), FEATURE_COLS, n_folds=5, seed=0)
    corrected = fdr_correct_reports([noise, planted])
    planted_c = max(corrected, key=lambda r: r.oos_auc)
    noise_c = min(corrected, key=lambda r: r.oos_auc)
    assert planted_c.p_value_fdr <= 0.10
    assert noise_c.p_value_fdr > 0.10
    assert planted_c.p_value_fdr < noise_c.p_value_fdr


def test_too_few_events_returns_honest_null():
    """A frame too small to validate returns a null report, not a fabricated edge."""
    small = _frame(3, 2.5).head(8)
    rep = walk_forward_validate(small, FEATURE_COLS, n_folds=5, seed=0)
    assert rep.n_folds == 0
    assert rep.p_value == 1.0
    assert rep.oos_net_expectancy_r == 0.0
