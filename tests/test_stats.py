"""
Statistical battery: each estimator behaves correctly on constructed data with a
known answer — significant when there's signal, null when there isn't, and the
overfitting probability (PBO) high for noise, low for a genuine edge.
"""

from __future__ import annotations

import numpy as np

from engine.forward.stats import (
    binomial_winrate,
    block_bootstrap_p,
    bootstrap_ci,
    cohens_d,
    deflated_sharpe_ratio,
    information_coefficient,
    mann_whitney,
    newey_west_t,
    permutation_test,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe,
    welch_t,
)


def test_two_sample_tests_separate_signal_from_null():
    rng = np.random.default_rng(0)
    strat = rng.normal(0.5, 1.0, 400)
    base = rng.normal(0.0, 1.0, 400)
    assert welch_t(strat, base)[1] < 0.01
    assert mann_whitney(strat, base) < 0.01
    assert cohens_d(strat, base) > 0.3
    # null: same distribution -> not significant
    a = rng.normal(0, 1, 400)
    b = rng.normal(0, 1, 400)
    assert welch_t(a, b)[1] > 0.05


def test_binomial_winrate():
    assert binomial_winrate(60, 100, 0.40) < 0.01  # 60% vs 40% base
    assert binomial_winrate(41, 100, 0.40) > 0.30  # ~base -> not significant


def test_permutation_test_detects_selection_skill():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 1, 600)
    scores = r + rng.normal(0, 0.3, 600)  # scores informative about r
    assert permutation_test(scores, r, thr=0.8, n=1000) < 0.05
    noise = rng.normal(0, 1, 600)  # uninformative scores
    assert permutation_test(noise, r, thr=0.8, n=1000) > 0.10


def test_block_bootstrap_and_newey_west():
    rng = np.random.default_rng(2)
    pos = rng.normal(0.3, 1.0, 500)
    assert block_bootstrap_p(pos, baseline=0.0, n=2000) < 0.05
    t, p = newey_west_t(pos, lags=5)
    assert t > 0 and p < 0.05
    null = rng.normal(0.0, 1.0, 500)
    assert block_bootstrap_p(null, baseline=0.0, n=2000) > 0.10
    lo, hi = bootstrap_ci(pos)
    assert lo < pos.mean() < hi


def test_sharpe_psr_and_deflation():
    rng = np.random.default_rng(3)
    r = rng.normal(0.1, 1.0, 500)
    assert sharpe(r) > 0
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert 0.0 <= psr <= 1.0
    # deflation: more trials searched -> lower confidence the edge is real
    dsr_few = deflated_sharpe_ratio(r, n_trials=2, trials_sr_variance=0.5)
    dsr_many = deflated_sharpe_ratio(r, n_trials=500, trials_sr_variance=0.5)
    assert dsr_many <= dsr_few


def test_information_coefficient():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 1, 500)
    scores = 2 * r + rng.normal(0, 0.5, 500)
    ic, p = information_coefficient(scores, r)
    assert ic > 0.5 and p < 0.01


def test_pbo_low_for_real_edge_high_for_noise():
    rng = np.random.default_rng(5)
    T, N = 600, 8
    # Real edge: strategy 0 has a genuine positive mean; rest are noise.
    M_real = rng.normal(0, 1, (T, N))
    M_real[:, 0] += 0.4
    res_real = probability_of_backtest_overfitting(M_real, n_splits=10)
    # Pure noise: no strategy is truly best -> selection overfits.
    M_noise = rng.normal(0, 1, (T, N))
    res_noise = probability_of_backtest_overfitting(M_noise, n_splits=10)
    assert res_real.pbo < 0.2
    assert res_noise.pbo > res_real.pbo
    assert res_real.median_oos_rank > 0.5
