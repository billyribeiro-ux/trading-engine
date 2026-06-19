"""
Model robustness to heterogeneous (NaN) feature matrices.

Real-data bug (found 2026-06-18): event feature vectors are HETEROGENEOUS -- a
vwap event has no leg_* features, a leg event has no level_* features -- so the
stacked training frame is full of NaN by construction. The original StandardScaler
used plain mean/std, so the logistic trained on NaN, predict_proba returned
all-NaN, and the walk-forward harness silently took ZERO signals while reporting a
garbage AUC (~0.83 from sorting a NaN array). The offline validate tests missed it
because synthetic frames have no NaN -- exactly the "synthetic hides
microstructure bugs" trap.

These lock the fix: NaN features are mean-imputed (neutral), predict_proba is
always finite, and a planted edge in a NaN-laden frame is actually DETECTED with
signals taken (not silently zeroed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.ml.model import StandardScaler, make_model
from engine.ml.validate import walk_forward_validate

FEATURE_COLS = ["f_signal", "f_typeA", "f_typeB"]
N = 800


def test_scaler_imputes_nan_to_zero_after_standardizing():
    X = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 20.0]])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    assert np.isfinite(Xs).all()
    # The imputed NaN becomes 0.0 == the standardized column mean (neutral).
    assert Xs[0, 1] == 0.0


def test_predict_proba_finite_with_nan_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    # Knock out type-specific columns for disjoint row subsets (mixed event types).
    X[:100, 1] = np.nan
    X[100:, 2] = np.nan
    y = (X[:, 0] + rng.normal(0, 0.5, size=200) > 0).astype(int)
    m = make_model("logistic", names=FEATURE_COLS).fit(X, y)
    p = m.predict_proba(X)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_all_nan_column_does_not_break_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    X[:, 2] = np.nan  # an entire feature missing for every row
    y = (X[:, 0] > 0).astype(int)
    m = make_model("logistic", names=FEATURE_COLS).fit(X, y)
    assert np.isfinite(m.predict_proba(X)).all()


def _nan_frame(seed: int) -> pd.DataFrame:
    """Planted edge in f_signal; f_typeA/f_typeB are type-specific and NaN for
    disjoint halves of the rows -- the exact shape of a real mixed-event frame."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=N)
    prob = 1.0 / (1.0 + np.exp(-2.5 * latent))
    y_win = rng.binomial(1, prob)
    f_signal = latent + rng.normal(0.0, 0.5, size=N)
    f_a = rng.normal(size=N)
    f_b = rng.normal(size=N)
    f_a[: N // 2] = np.nan  # "type B" rows lack the type-A feature
    f_b[N // 2 :] = np.nan  # "type A" rows lack the type-B feature
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": pd.Timestamp("2026-06-01"),
            "event_index": np.arange(N),
            "f_signal": f_signal,
            "f_typeA": f_a,
            "f_typeB": f_b,
            "y_win": y_win.astype(int),
            "y_bracket_r": np.where(y_win == 1, 2.0, -1.0).astype(float),
        }
    )


def test_planted_edge_detected_through_nan_columns():
    """The real regression: a NaN-laden frame must NOT silently zero signals.
    Old code -> predict_proba NaN -> 0 signals -> fake null. Fixed -> signals
    taken and the edge is detected."""
    rep = walk_forward_validate(_nan_frame(2), FEATURE_COLS, n_folds=5, seed=0)
    assert rep.n_total_signals > 0, "NaN features silently zeroed all signals (bug)"
    assert rep.p_value < 0.05, f"planted edge missed through NaN columns (p={rep.p_value})"
    assert rep.oos_auc > 0.60, rep.oos_auc
    assert rep.oos_net_expectancy_r > 0.0


def test_gbt_model_is_finite_handles_nan_and_single_class():
    """The optional GBT (sklearn HistGBT) consumes NaN natively, returns finite
    probabilities, and degrades gracefully on a single-class fit."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[:100, 1] = np.nan  # heterogeneous (type-specific) features
    y = (X[:, 0] + rng.normal(0, 0.5, size=200) > 0).astype(int)
    m = make_model("gbt", names=FEATURE_COLS).fit(X, y)
    p = m.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert m.feature_names == FEATURE_COLS
    # single-class training must not raise.
    m2 = make_model("gbt").fit(X, np.ones(200, dtype=int))
    assert np.isfinite(m2.predict_proba(X)).all()
