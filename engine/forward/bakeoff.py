"""
Version bake-off: rank candidate (model + feature-set) versions by FORWARD
evidence, not in-sample fit.

Searching over model classes, feature subsets, and thresholds inflates false
positives, so a version is PROMOTED only if it BOTH persists out-of-time
(forward_test: realized edge > 0, significant, enough holdout signals) AND
survives Benjamini-Hochberg across every version tried. "Nothing promoted" is a
valid, common, honest result — it means no version beat noise forward. This is the
opposite of chasing "100% accuracy": we keep only what hard out-of-time evidence
supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..intraday.multiple_testing import benjamini_hochberg
from .runner import ForwardTestResult, forward_test


@dataclass(frozen=True)
class ModelVersion:
    name: str
    model_kind: str = "logistic"  # 'logistic' | 'gbt'
    features: tuple[str, ...] | None = None  # None -> all f_* columns in the frame
    proba_threshold: float = 0.55


@dataclass(frozen=True)
class VersionResult:
    version: str
    model_kind: str
    result: ForwardTestResult
    p_value_fdr: float  # realized forward p, BH-adjusted across versions
    promoted: bool


def default_versions() -> list[ModelVersion]:
    """A reasonable starting slate: the numpy logistic and the GBT, full features."""
    return [
        ModelVersion("logistic", "logistic"),
        ModelVersion("gbt", "gbt"),
    ]


def bake_off(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    versions: Sequence[ModelVersion] | None = None,
    *,
    horizon_bars: int = 24,
    cutoff_frac: float = 0.7,
    cost_r: float = 0.05,
    fdr: float = 0.10,
    min_holdout_signals: int = 10,
    seed: int = 0,
) -> list[VersionResult]:
    """Forward-test each version, FDR-correct across versions on the realized
    (forward) p-value, and rank — promoted first."""
    versions = list(versions) if versions is not None else default_versions()
    pairs: list[tuple[ModelVersion, ForwardTestResult]] = []
    for v in versions:
        fcs = [c for c in (v.features or feature_cols) if c in df.columns]
        r = forward_test(
            df,
            fcs,
            horizon_bars=horizon_bars,
            cutoff_frac=cutoff_frac,
            proba_threshold=v.proba_threshold,
            cost_r=cost_r,
            min_holdout_signals=min_holdout_signals,
            model_kind=v.model_kind,
            seed=seed,
        )
        pairs.append((v, r))

    pvals = [r.realized_p for _, r in pairs]
    flags = benjamini_hochberg(pvals, fdr=fdr) if pvals else []
    p_adj = _bh_adjust(pvals)

    out = [
        VersionResult(
            version=v.name,
            model_kind=v.model_kind,
            result=r,
            p_value_fdr=padj,
            promoted=bool(sig and r.persisted),
        )
        for (v, r), padj, sig in zip(pairs, p_adj, flags)
    ]
    out.sort(key=lambda x: (not x.promoted, x.p_value_fdr, -x.result.realized_edge_r))
    return out


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values, original order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = prev
    return adj


def render_bakeoff(rows: Sequence[VersionResult]) -> str:
    promoted = [r for r in rows if r.promoted]
    lines = [
        f"Bake-off: {len(rows)} versions, {len(promoted)} promoted "
        f"(persist forward + survive FDR).",
        f"  {'version':<14}{'model':<10}{'valid_R':>9}{'real_R':>8}{'decay':>8}"
        f"{'hAUC':>6}{'sig':>5}{'p_fdr':>8}  promoted",
    ]
    for r in rows:
        t = r.result
        lines.append(
            f"  {r.version:<14}{r.model_kind:<10}{t.validated_edge_r:>+9.3f}"
            f"{t.realized_edge_r:>+8.3f}{t.forward_decay_r:>+8.3f}{t.holdout_auc:>6.2f}"
            f"{t.n_holdout_signals:>5}{r.p_value_fdr:>8.3f}  {r.promoted}"
        )
    if not promoted:
        lines.append("  -> no version persists forward. Honest null; do not trade.")
    return "\n".join(lines)
