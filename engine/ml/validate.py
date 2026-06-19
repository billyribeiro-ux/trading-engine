"""
Walk-forward validation harness for the learning layer.

This is the piece that decides whether a discovered pattern is REAL or noise.
Without it, a model that finds spurious structure in-sample looks like a money
machine and dies live. It enforces, in order:

  1. WALK-FORWARD: split the time-ordered events into sequential folds; always
     train on the past, test on the future. No shuffling -- shuffling lets the
     model peek at the regime it's being tested in.

  2. PURGE + EMBARGO: a label looks forward up to `horizon` / bracket max_bars.
     Any training event whose label window overlaps the test fold is PURGED
     (its outcome partly depends on test-period bars). An additional EMBARGO of
     a few bars after each test fold keeps adjacent leakage out. This is the
     Lopez de Prado discipline adapted to intraday events.

  3. COST-AWARE OUTCOME: edge is measured on outcomes NET of the cost model
     (spread + slippage + commission), never gross. A gross edge that costs eat
     is not an edge.

  4. MULTIPLE-TESTING: when many configurations/features are screened, raw
     p-values overstate significance. Benjamini-Hochberg FDR control (shared with
     the intraday layer) is applied so surviving signals aren't just the luckiest
     of many tries.

The output is an honest out-of-sample performance summary per signal config:
net expectancy, hit rate, fold-by-fold decay, and an FDR-corrected p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..intraday.multiple_testing import benjamini_hochberg
from .model import SignalModel, make_model


@dataclass(frozen=True)
class FoldReport:
    fold: int
    n_train: int
    n_test: int
    test_auc: float
    test_net_expectancy_r: float  # mean NET R of taken signals in this fold
    n_signals: int  # events the model flagged (proba >= threshold)
    hit_rate: float


@dataclass(frozen=True)
class ValidationReport:
    symbol: str
    n_events: int
    n_folds: int
    oos_net_expectancy_r: float  # pooled out-of-sample net R per signal
    oos_hit_rate: float
    oos_auc: float
    p_value: float  # bootstrap p-value of edge > 0
    p_value_fdr: float  # after Benjamini-Hochberg (set by caller)
    folds: tuple[FoldReport, ...]
    decay: float  # first-fold minus last-fold expectancy
    n_total_signals: int


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC without sklearn (rank statistic). 0.5 if degenerate."""
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Mann-Whitney U / (n_pos*n_neg)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _apply_cost(r: np.ndarray, cost_r: float) -> np.ndarray:
    """Subtract per-trade cost (in R units) from realized R."""
    return r - cost_r


def _bootstrap_p(net_r: np.ndarray, n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided bootstrap p-value that mean net R <= 0."""
    if net_r.size < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = net_r.mean()
    if obs <= 0:
        return 1.0
    centered = net_r - obs
    count = 0
    for _ in range(n_boot):
        sample = rng.choice(centered, size=net_r.size, replace=True)
        if sample.mean() >= obs:
            count += 1
    return (count + 1) / (n_boot + 1)


def _bootstrap_edge_p(
    taken_r: np.ndarray, baseline_mean: float, n_boot: int = 2000, seed: int = 0
) -> float:
    """
    One-sided bootstrap p-value that the model's selected-signal mean R is no
    better than the BASELINE mean R (taking all events).

    This is the decisive correction: a profitable bracket (e.g. 2:1 reward/risk
    at a 50% base rate) is positive-expectancy by construction, with NO skill.
    Measuring raw expectancy credits the bracket's geometry as if it were edge.
    The real question is whether the model's SELECTION beats the base rate. We
    test mean(taken) - baseline_mean > 0.
    """
    if taken_r.size < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = taken_r.mean() - baseline_mean
    if obs <= 0:
        return 1.0
    centered = taken_r - taken_r.mean()  # null: taken has same mean as baseline
    count = 0
    for _ in range(n_boot):
        sample = rng.choice(centered, size=taken_r.size, replace=True)
        if sample.mean() >= obs:
            count += 1
    return (count + 1) / (n_boot + 1)


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "y_win",
    r_col: str = "y_bracket_r",
    n_folds: int = 5,
    embargo_bars: int = 3,
    horizon_bars: int = 24,
    proba_threshold: float = 0.55,
    cost_r: float = 0.05,
    model_kind: str = "logistic",
    seed: int = 0,
) -> ValidationReport:
    """
    Run purged, embargoed walk-forward validation on a labeled event frame.

    df must be time-ordered (date, event_index) -- build_training_frame does
    this. For each of `n_folds` sequential test blocks, train on all PRIOR rows
    (minus purged/embargoed ones), predict on the test block, take signals where
    predicted proba >= threshold, and measure NET R. Pools out-of-sample results
    and returns an honest ValidationReport.

    cost_r is the round-trip cost in R units (so a 0.05 means each trade starts
    0.05R in the hole). Defaults are conservative.
    """
    if df.empty or len(df) < n_folds * 4:
        return ValidationReport(
            symbol=str(df["symbol"].iloc[0]) if "symbol" in df and len(df) else "?",
            n_events=len(df),
            n_folds=0,
            oos_net_expectancy_r=0.0,
            oos_hit_rate=0.0,
            oos_auc=0.5,
            p_value=1.0,
            p_value_fdr=1.0,
            folds=(),
            decay=0.0,
            n_total_signals=0,
        )

    df = df.reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    r = df[r_col].to_numpy(dtype=float)
    # Row order IS the clock: rows are time-sorted, so purge/embargo below
    # operate directly on integer row indices.
    fold_bounds = np.linspace(0, len(df), n_folds + 1, dtype=int)
    folds: list[FoldReport] = []
    pooled_net_r: list[float] = []  # NET R of model-selected signals
    pooled_baseline_r: list[float] = []  # NET R of ALL test events (no skill)
    pooled_y: list[int] = []
    pooled_p: list[float] = []

    for k in range(n_folds):
        test_lo, test_hi = fold_bounds[k], fold_bounds[k + 1]
        if test_hi - test_lo < 2:
            continue
        test_idx = np.arange(test_lo, test_hi)
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:test_lo] = True
        purge_cut = test_lo - embargo_bars
        for i in np.where(train_mask)[0]:
            if i + horizon_bars >= purge_cut:
                train_mask[i] = False
        if train_mask.sum() < 8:
            continue

        model: SignalModel = make_model(model_kind, names=feature_cols)
        model.fit(X[train_mask], y[train_mask])
        p_test = model.predict_proba(X[test_idx])

        auc = _auc(y[test_idx], p_test)
        # Baseline: net R of taking EVERY event in the test fold (no skill).
        baseline_net = _apply_cost(r[test_idx], cost_r)
        pooled_baseline_r.extend(baseline_net.tolist())

        take = p_test >= proba_threshold
        if take.sum() > 0:
            net = _apply_cost(r[test_idx][take], cost_r)
            exp_r = float(net.mean())
            hit = float((net > 0).mean())
            pooled_net_r.extend(net.tolist())
        else:
            exp_r, hit = 0.0, 0.0
        pooled_y.extend(y[test_idx].tolist())
        pooled_p.extend(p_test.tolist())

        folds.append(
            FoldReport(
                fold=k,
                n_train=int(train_mask.sum()),
                n_test=len(test_idx),
                test_auc=auc,
                test_net_expectancy_r=exp_r,
                n_signals=int(take.sum()),
                hit_rate=hit,
            )
        )

    net_arr = np.array(pooled_net_r, dtype=float)
    baseline_arr = np.array(pooled_baseline_r, dtype=float)
    baseline_mean = float(baseline_arr.mean()) if baseline_arr.size else 0.0
    oos_exp = float(net_arr.mean()) if net_arr.size else 0.0
    oos_hit = float((net_arr > 0).mean()) if net_arr.size else 0.0
    oos_auc = _auc(np.array(pooled_y), np.array(pooled_p)) if pooled_y else 0.5
    # Edge p-value: does model selection beat the base rate (not just > 0)?
    pval = _bootstrap_edge_p(net_arr, baseline_mean, seed=seed)
    decay = (
        (folds[0].test_net_expectancy_r - folds[-1].test_net_expectancy_r)
        if len(folds) >= 2
        else 0.0
    )

    return ValidationReport(
        symbol=str(df["symbol"].iloc[0]) if "symbol" in df else "?",
        n_events=len(df),
        n_folds=len(folds),
        oos_net_expectancy_r=oos_exp - baseline_mean,  # EDGE over baseline
        oos_hit_rate=oos_hit,
        oos_auc=oos_auc,
        p_value=pval,
        p_value_fdr=pval,
        folds=tuple(folds),
        decay=decay,
        n_total_signals=int(net_arr.size),
    )


def fdr_correct_reports(reports: list[ValidationReport]) -> list[ValidationReport]:
    """
    Apply Benjamini-Hochberg across multiple validated configs.

    When several brackets/thresholds/symbols are screened, correct their p-values
    so survivors aren't just the luckiest of many tries. Returns new reports with
    p_value_fdr populated.
    """
    if not reports:
        return reports
    pvals = [r.p_value for r in reports]
    rejected = benjamini_hochberg(pvals)
    # benjamini_hochberg returns significance flags; derive a simple step-up
    # adjusted p-value for reporting (BH-adjusted), preserving original order.
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    p_adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        p_adj[i] = val
        prev = val
    out = []
    for r, pa, _rej in zip(reports, p_adj, rejected):
        out.append(
            ValidationReport(
                symbol=r.symbol,
                n_events=r.n_events,
                n_folds=r.n_folds,
                oos_net_expectancy_r=r.oos_net_expectancy_r,
                oos_hit_rate=r.oos_hit_rate,
                oos_auc=r.oos_auc,
                p_value=r.p_value,
                p_value_fdr=float(pa),
                folds=r.folds,
                decay=r.decay,
                n_total_signals=r.n_total_signals,
            )
        )
    return out
