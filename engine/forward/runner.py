"""
Forward testing: does the validated edge actually hold out-of-TIME?

Walk-forward validation tells you a config's honest OOS edge on the lookback. But
the real question for a live system is whether that edge PERSISTS on data the
whole process never touched. This runner answers it on history without waiting:

  1. Split the labeled frame by time at `cutoff_frac` into TRAIN and HOLDOUT, with
     a purge band so train labels don't reach into the holdout period.
  2. Walk-forward-validate on TRAIN -> the edge the process "promised".
  3. Fit the final model on TRAIN only, then score the HOLDOUT (events the model
     has never seen) and measure the REALIZED net-R over the holdout baseline.
  4. Report validated vs realized, and the FORWARD DECAY between them.

A config whose train edge is strong but whose holdout edge collapses is overfit —
exactly what forward testing is for. `persisted` is the honest verdict: realized
edge > 0, significant, on enough holdout signals. Everything is net of cost and
measured OVER baseline (taking all holdout events), never raw bracket profit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ml.model import make_model
from ..ml.validate import _apply_cost, _auc, _bootstrap_edge_p, walk_forward_validate


@dataclass(frozen=True)
class ForwardTestResult:
    symbol: str
    n_train: int
    n_holdout: int
    n_holdout_signals: int
    validated_edge_r: float  # train walk-forward edge over baseline
    validated_p: float
    realized_edge_r: float  # holdout: taken mean net-R minus holdout baseline
    realized_hit_rate: float
    realized_p: float  # bootstrap: holdout taken > holdout baseline
    holdout_auc: float
    forward_decay_r: float  # validated_edge_r - realized_edge_r (overfit gauge)
    persisted: bool


def forward_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "y_win",
    r_col: str = "y_bracket_r",
    horizon_bars: int = 24,
    cutoff_frac: float = 0.7,
    proba_threshold: float = 0.55,
    cost_r: float = 0.05,
    min_holdout_signals: int = 10,
    n_folds: int = 5,
    model_kind: str = "logistic",
    seed: int = 0,
) -> ForwardTestResult:
    """Out-of-time forward test on a labeled, time-ordered event frame."""
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df and len(df) else "?"
    empty = ForwardTestResult(symbol, len(df), 0, 0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, False)
    if df.empty or len(df) < max(40, n_folds * 8):
        return empty

    df = df.reset_index(drop=True)
    cut = int(len(df) * cutoff_frac)
    # Purge: train ends `horizon_bars` rows before the cut so train labels (which
    # look forward up to the bracket horizon) don't reach into the holdout period.
    train_end = max(0, cut - horizon_bars)
    train = df.iloc[:train_end]
    holdout = df.iloc[cut:]
    if len(train) < n_folds * 8 or len(holdout) < min_holdout_signals:
        return empty

    # 1) What walk-forward promised on the past.
    report = walk_forward_validate(
        train,
        feature_cols,
        label_col=label_col,
        r_col=r_col,
        n_folds=n_folds,
        horizon_bars=horizon_bars,
        proba_threshold=proba_threshold,
        cost_r=cost_r,
        model_kind=model_kind,
        seed=seed,
    )

    # 2) Fit final model on train, score the unseen holdout.
    model = make_model(model_kind, names=list(feature_cols))
    model.fit(train[feature_cols].to_numpy(dtype=float), train[label_col].to_numpy(dtype=int))
    Xh = holdout[feature_cols].to_numpy(dtype=float)
    yh = holdout[label_col].to_numpy(dtype=int)
    rh = holdout[r_col].to_numpy(dtype=float)
    p = model.predict_proba(Xh)

    holdout_baseline = float(_apply_cost(rh, cost_r).mean()) if rh.size else 0.0
    take = p >= proba_threshold
    taken_net = _apply_cost(rh[take], cost_r) if take.any() else np.array([], dtype=float)
    realized_edge = (float(taken_net.mean()) - holdout_baseline) if taken_net.size else 0.0
    realized_hit = float((taken_net > 0).mean()) if taken_net.size else 0.0
    realized_p = _bootstrap_edge_p(taken_net, holdout_baseline, seed=seed)
    holdout_auc = _auc(yh, p)

    persisted = taken_net.size >= min_holdout_signals and realized_edge > 0.0 and realized_p < 0.10
    return ForwardTestResult(
        symbol=symbol,
        n_train=len(train),
        n_holdout=len(holdout),
        n_holdout_signals=int(taken_net.size),
        validated_edge_r=report.oos_net_expectancy_r,
        validated_p=report.p_value,
        realized_edge_r=realized_edge,
        realized_hit_rate=realized_hit,
        realized_p=realized_p,
        holdout_auc=holdout_auc,
        forward_decay_r=report.oos_net_expectancy_r - realized_edge,
        persisted=bool(persisted),
    )
