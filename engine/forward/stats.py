"""
The full statistical battery — every test that can confirm or break an edge.

Edge-over-baseline and a single bootstrap p are necessary, not sufficient. A
distinguished quant throws the whole toolkit at a candidate before believing it:
parametric and non-parametric significance, serial-correlation-robust inference,
Sharpe-based tests that PENALISE for the number of strategies tried, and the
backtest-overfitting probability (CSCV/PBO). Each function here is a standalone,
unit-tested estimator on a per-trade R-multiple series.

Conventions: `r` is a 1-D array of realised per-trade R-multiples (net of cost).
"Edge over baseline" means the strategy's mean minus a reference mean (taking all
events). p-values are one-sided (is the strategy BETTER), unless noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------- #
# Parametric & non-parametric two-sample tests (strategy vs baseline)
# ----------------------------------------------------------------------------- #
def welch_t(strategy: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    """Welch's t (unequal variance), one-sided p that strategy mean > baseline."""
    if len(strategy) < 2 or len(baseline) < 2:
        return 0.0, 1.0
    t, p_two = stats.ttest_ind(strategy, baseline, equal_var=False)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(t), float(p_one)


def mann_whitney(strategy: np.ndarray, baseline: np.ndarray) -> float:
    """Mann-Whitney U one-sided p (strategy stochastically greater)."""
    if not len(strategy) or not len(baseline):
        return 1.0
    return float(stats.mannwhitneyu(strategy, baseline, alternative="greater").pvalue)


def binomial_winrate(wins: int, n: int, base_rate: float) -> float:
    """One-sided binomial p that the win rate exceeds base_rate."""
    if n == 0:
        return 1.0
    return float(stats.binomtest(wins, n, base_rate, alternative="greater").pvalue)


def cohens_d(strategy: np.ndarray, baseline: np.ndarray) -> float:
    if len(strategy) < 2 or len(baseline) < 2:
        return 0.0
    nx, ny = len(strategy), len(baseline)
    sp = math.sqrt(
        ((nx - 1) * strategy.var(ddof=1) + (ny - 1) * baseline.var(ddof=1)) / (nx + ny - 2)
    )
    return float((strategy.mean() - baseline.mean()) / sp) if sp > 0 else 0.0


# ----------------------------------------------------------------------------- #
# Resampling: permutation + block bootstrap (serial-correlation aware)
# ----------------------------------------------------------------------------- #
def permutation_test(
    scores: np.ndarray, r: np.ndarray, thr: float, n: int = 2000, seed: int = 0
) -> float:
    """Shuffle the score<->return pairing; p = fraction of shuffles whose taken
    mean >= observed. Tests whether the model's SELECTION carries information,
    making no distributional assumption."""
    scores = np.asarray(scores, float)
    r = np.asarray(r, float)
    take = scores >= thr
    if take.sum() == 0:
        return 1.0
    observed = r[take].mean()
    rng = np.random.default_rng(seed)
    k = int(take.sum())
    ge = 0
    for _ in range(n):
        idx = rng.choice(len(r), size=k, replace=False)
        if r[idx].mean() >= observed:
            ge += 1
    return (ge + 1) / (n + 1)


def block_bootstrap_p(
    r: np.ndarray, baseline: float = 0.0, block: int = 5, n: int = 5000, seed: int = 0
) -> float:
    """Stationary/circular block bootstrap p that mean(r) > baseline. Blocks keep
    local serial correlation, so the p is not inflated by autocorrelated returns
    (the iid bootstrap's blind spot)."""
    r = np.asarray(r, float)
    m = len(r)
    if m < 2:
        return 1.0
    obs = r.mean() - baseline
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    means = np.empty(n)
    centred = r - r.mean()  # resample under H0: mean == baseline
    for i in range(n):
        starts = rng.integers(0, m, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % m
        means[i] = centred[idx[:m]].mean()
    return float((means >= obs).mean())


def bootstrap_ci(
    r: np.ndarray, alpha: float = 0.05, n: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    r = np.asarray(r, float)
    if len(r) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot = r[rng.integers(0, len(r), size=(n, len(r)))].mean(axis=1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


# ----------------------------------------------------------------------------- #
# Serial-correlation-robust mean t-stat (Newey-West / HAC)
# ----------------------------------------------------------------------------- #
def newey_west_t(r: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """t-stat for mean(r) != 0 with Newey-West (HAC) standard error. One-sided p."""
    r = np.asarray(r, float)
    n = len(r)
    if n < 3:
        return 0.0, 1.0
    x = r - r.mean()
    gamma0 = (x @ x) / n
    s = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1)  # Bartlett kernel
        cov = (x[lag:] @ x[:-lag]) / n
        s += 2 * w * cov
    se = math.sqrt(max(s, 1e-18) / n)
    t = r.mean() / se if se > 0 else 0.0
    p_one = 1 - stats.norm.cdf(t)
    return float(t), float(p_one)


# ----------------------------------------------------------------------------- #
# Sharpe-based tests that penalise for multiple trials
# ----------------------------------------------------------------------------- #
def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado): P(true SR > benchmark) given sample SR,
    length, skew and kurtosis. Corrects the Sharpe for non-normal returns."""
    r = np.asarray(r, float)
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe(r)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # non-excess
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(r: np.ndarray, n_trials: int, trials_sr_variance: float) -> float:
    """DSR: PSR against a benchmark SR inflated for the NUMBER of strategies tried.
    Directly answers 'is the best of N searched strategies real?'. trials_sr_variance
    is the variance of the SRs across the trials searched."""
    r = np.asarray(r, float)
    n = len(r)
    if n < 3 or n_trials < 1:
        return 0.0
    emc = 0.5772156649  # Euler-Mascheroni
    # expected max of N standard-normal SRs (Bailey-LdP approximation)
    e_max = math.sqrt(max(trials_sr_variance, 1e-12)) * (
        (1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
        + emc * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    )
    return probabilistic_sharpe_ratio(r, sr_benchmark=e_max)


# ----------------------------------------------------------------------------- #
# Information coefficient (rank corr of score vs forward return)
# ----------------------------------------------------------------------------- #
def information_coefficient(scores: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Spearman IC between model score and realised return, with its p-value."""
    scores = np.asarray(scores, float)
    r = np.asarray(r, float)
    if len(scores) < 3:
        return 0.0, 1.0
    res = stats.spearmanr(scores, r)
    return float(res.statistic), float(res.pvalue)


# ----------------------------------------------------------------------------- #
# Backtest-overfitting capstone: CSCV / PBO (López de Prado)
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PBOResult:
    pbo: float  # probability of backtest overfitting (best IS -> below-median OOS)
    n_combos: int
    median_oos_rank: float  # of the IS-best strategy (0..1; >0.5 good)


def probability_of_backtest_overfitting(
    M: np.ndarray, n_splits: int = 10, seed: int = 0
) -> PBOResult:
    """CSCV PBO over a (T observations x N strategies) per-observation return matrix.

    Split T into n_splits blocks; for every balanced split into IS/OOS halves,
    pick the IS-best strategy and find its OOS performance RANK among all
    strategies. PBO = fraction of splits where the IS-best ranks below the OOS
    median (logit < 0). High PBO => the selection process overfits."""
    from itertools import combinations

    M = np.asarray(M, float)
    T, N = M.shape
    if N < 2 or n_splits < 2 or n_splits % 2 or T < n_splits:
        return PBOResult(1.0, 0, 0.0)
    blocks = np.array_split(np.arange(T), n_splits)
    logits, ranks = [], []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_idx])
        is_perf = np.array([sharpe(M[is_rows, j]) for j in range(N)])
        oos_perf = np.array([sharpe(M[oos_rows, j]) for j in range(N)])
        best = int(np.argmax(is_perf))
        # OOS rank of the IS-best (fraction of strategies it beats OOS)
        w = (oos_perf < oos_perf[best]).mean()
        ranks.append(w)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return PBOResult(
        pbo=float((logits < 0).mean()),
        n_combos=len(logits),
        median_oos_rank=float(np.median(ranks)),
    )
