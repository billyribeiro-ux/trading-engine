"""
Multiple-testing correction for scenario selection.

The ensemble runs many scenarios (anchors x triggers x parameters). If you test
36 scenarios at the 5% level, you expect ~1.8 to look "significant" by pure
chance even if none has real edge. Picking the best-looking scenario without
correcting for the search is the single most common way a quant backtest fools
its author -- the "best" is partly luck, and the more you searched, the luckier
the winner.

This module deflates for the search:

    * Per-scenario significance: one-sample test that net expectancy > 0, using a
      bootstrap p-value (distribution-free; net-R is skewed and fat-tailed so a
      t-test understates tail risk).
    * Benjamini-Hochberg FDR: controls the expected false-discovery rate across
      all scenarios -- the right tool when you want to keep several edges and
      bound how many are flukes (vs Bonferroni, which controls family-wise error
      and is far stricter / better when any false positive is costly).
    * Bonferroni: reported alongside as the conservative bound.

A scenario that survives BH at a chosen FDR is one whose OOS edge is unlikely to
be an artifact of having searched many scenarios. That is the set worth trading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioTest:
    scenario: str
    side: str
    n: int
    mean_net_r: float
    p_value: float  # P(mean <= 0) under the bootstrap null
    bh_significant: bool  # survives Benjamini-Hochberg at fdr
    bonferroni_significant: bool  # survives Bonferroni at alpha
    bh_threshold: float  # the BH critical value applied
    rank: int  # ascending p-value rank (1 = smallest p)


def _bootstrap_p_value(net_r: np.ndarray, n_boot: int = 10_000, seed: int = 17) -> float:
    """
    Bootstrap p-value for H0: mean(net_r) <= 0 vs H1: mean(net_r) > 0.

    We recentre the sample to the null (mean 0), resample, and ask how often the
    resampled mean reaches the observed mean. Equivalent to a one-sided bootstrap
    test of positive expectancy. Distribution-free -- no normality assumption.
    """
    net_r = np.asarray(net_r, dtype=float)
    net_r = net_r[~np.isnan(net_r)]
    n = net_r.size
    if n < 5:
        return 1.0
    observed = net_r.mean()
    if observed <= 0:
        return 1.0
    centered = net_r - observed  # null: mean 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = centered[idx].mean(axis=1)
    # p = fraction of null resamples whose mean >= observed.
    p = float((boot_means >= observed).mean())
    # Guard against p==0 (finite resampling) with the standard +1 correction.
    return max(p, 1.0 / (n_boot + 1))


def benjamini_hochberg(p_values: list[float], fdr: float = 0.10) -> list[bool]:
    """
    Benjamini-Hochberg step-up procedure. Returns a boolean per input p-value
    (in original order) indicating significance at the given FDR.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    # Largest k such that p_(k) <= (k/m) * fdr.
    thresholds = (np.arange(1, m + 1) / m) * fdr
    below = sorted_p <= thresholds
    if not below.any():
        sig_sorted = np.zeros(m, dtype=bool)
    else:
        k_max = np.max(np.where(below)[0])
        sig_sorted = np.arange(m) <= k_max
    # Map back to original order.
    sig = np.zeros(m, dtype=bool)
    sig[order] = sig_sorted
    return sig.tolist()


def correct_scenarios(
    scenario_net_r: dict[tuple[str, str], np.ndarray],
    fdr: float = 0.10,
    alpha: float = 0.05,
) -> list[ScenarioTest]:
    """
    Apply multiple-testing correction across all scenarios.

    Parameters
    ----------
    scenario_net_r : {(scenario, side): array of per-trade net R}
        Use OUT-OF-SAMPLE net R from walk-forward when available -- correcting
        in-sample numbers corrects the wrong thing.
    fdr   : Benjamini-Hochberg false-discovery rate.
    alpha : Bonferroni family-wise alpha.

    Returns one ScenarioTest per scenario, sorted by p-value ascending.
    """
    keys = list(scenario_net_r.keys())
    p_values: list[float] = []
    means: list[float] = []
    ns: list[int] = []
    for k in keys:
        arr = np.asarray(scenario_net_r[k], dtype=float)
        p_values.append(_bootstrap_p_value(arr))
        means.append(float(np.nanmean(arr)) if arr.size else float("nan"))
        ns.append(int(arr.size))

    m = len(keys)
    bh_flags = benjamini_hochberg(p_values, fdr=fdr)
    bonf_alpha = alpha / m if m else alpha
    bonf_flags = [p <= bonf_alpha for p in p_values]

    order = np.argsort(p_values)
    rank_of = {int(idx): r + 1 for r, idx in enumerate(order)}

    tests: list[ScenarioTest] = []
    for i, k in enumerate(keys):
        rank = rank_of[i]
        bh_thresh = (rank / m) * fdr if m else float("nan")
        tests.append(
            ScenarioTest(
                scenario=k[0],
                side=k[1],
                n=ns[i],
                mean_net_r=means[i],
                p_value=p_values[i],
                bh_significant=bool(bh_flags[i]),
                bonferroni_significant=bool(bonf_flags[i]),
                bh_threshold=bh_thresh,
                rank=rank,
            )
        )
    tests.sort(key=lambda t: t.p_value)
    return tests
