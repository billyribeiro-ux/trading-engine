"""
What actually drives the forward edge? Permutation importance, the honest way.

A black-box edge you can't explain is one you can't trust. This measures
importance the way that matters: fit on train, then on the UNSEEN holdout, shuffle
one feature at a time and measure how much the REALIZED edge (and AUC) drops. A
feature whose destruction kills the forward edge is a real driver; one whose
shuffling changes nothing is decoration. Importance is on realized edge — not
in-sample fit — so it answers "what is the edge built on out-of-time?".

If the top drivers are sensible (mean-reversion, relative strength, trend) it
builds trust; if they're nonsensical, distrust the edge regardless of its p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ml.model import make_model
from ..ml.validate import _apply_cost, _auc


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    edge_drop: float  # mean drop in realized holdout edge when this feature is permuted
    auc_drop: float


def _edge_auc(model, X, y, r, thr, cost):
    p = model.predict_proba(X)
    baseline = float(_apply_cost(r, cost).mean()) if r.size else 0.0
    take = p >= thr
    taken = _apply_cost(r[take], cost) if take.any() else np.array([], dtype=float)
    edge = (float(taken.mean()) - baseline) if taken.size else 0.0
    return edge, _auc(y, p)


def permutation_importance(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    model_kind: str = "gbt",
    label_col: str = "y_win",
    r_col: str = "y_bracket_r",
    cutoff_frac: float = 0.7,
    horizon_bars: int = 8,
    proba_threshold: float = 0.55,
    cost_r: float = 0.05,
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[float, float, list[FeatureImportance]]:
    """Fit on the train split, then rank features by how much shuffling each in the
    holdout drops the realized edge. Returns (base_edge, base_auc, ranked)."""
    feature_cols = list(feature_cols)
    df = df.reset_index(drop=True)
    n = len(df)
    cut = int(n * cutoff_frac)
    train = df.iloc[: max(0, cut - horizon_bars)]
    holdout = df.iloc[cut:]
    if len(train) < 40 or len(holdout) < 20:
        return 0.0, 0.5, []

    model = make_model(model_kind, names=feature_cols)
    model.fit(train[feature_cols].to_numpy(dtype=float), train[label_col].to_numpy(dtype=int))
    Xh = holdout[feature_cols].to_numpy(dtype=float)
    yh = holdout[label_col].to_numpy(dtype=int)
    rh = holdout[r_col].to_numpy(dtype=float)

    base_edge, base_auc = _edge_auc(model, Xh, yh, rh, proba_threshold, cost_r)
    rng = np.random.default_rng(seed)
    out: list[FeatureImportance] = []
    for j, col in enumerate(feature_cols):
        e_drops, a_drops = [], []
        for _ in range(n_repeats):
            Xp = Xh.copy()
            rng.shuffle(Xp[:, j])  # destroy this feature's signal, keep its marginal
            e, a = _edge_auc(model, Xp, yh, rh, proba_threshold, cost_r)
            e_drops.append(base_edge - e)
            a_drops.append(base_auc - a)
        out.append(FeatureImportance(col, float(np.mean(e_drops)), float(np.mean(a_drops))))
    out.sort(key=lambda x: x.edge_drop, reverse=True)
    return base_edge, base_auc, out


def render_importance(base_edge: float, base_auc: float, rows: Sequence[FeatureImportance]) -> str:
    lines = [
        f"Forward feature importance (holdout edge={base_edge:+.3f}, auc={base_auc:.2f}):",
        f"  {'feature':<26}{'edge_drop':>11}{'auc_drop':>10}",
    ]
    for r in rows:
        lines.append(f"  {r.feature:<26}{r.edge_drop:>+11.3f}{r.auc_drop:>+10.3f}")
    return "\n".join(lines)
