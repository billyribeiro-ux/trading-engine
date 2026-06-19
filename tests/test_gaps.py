"""
Adversarial tests for engine.gaps.statistics.

Wilson score CI and beta-binomial (Bayesian) shrinkage carry the engine's
"don't let small samples lie" contract. These tests pin EXACT numeric values
(hand-derivable from the closed-form Wilson formula and reproduced once against
the implementation) and assert the structural invariants the docstrings promise:

  * Wilson bounds live in [0, 1] and lower <= upper, even at the 0/1 edges.
  * Wilson is symmetric for p = 0.5 (center sits exactly at 0.5).
  * A wider z gives a wider interval.
  * Shrinkage pulls a 3/3 = 100% cell hard toward the prior, and leaves a
    400-trial cell within a whisker of the raw rate.
  * estimate_proportion's sufficiency gate keys off BOTH n and interval width.

No network. No randomness in the asserted values (bootstrap is exercised only
with a fixed seed where its output is checked structurally, not pinned).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.gaps.statistics import (
    beta_binomial_posterior,
    bootstrap_mean_ci,
    estimate_proportion,
    wilson_interval,
)

_Z_95 = 1.959963984540054


def _wilson_reference(successes: int, trials: int, z: float = _Z_95):
    """Independent closed-form Wilson, derived from the standard formula, so the
    locked values below are not just 'whatever the engine emits'."""
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# Wilson confidence interval -- locked numeric values
# ---------------------------------------------------------------------------
def test_wilson_locked_values_50_of_100():
    """50/100 at 95%: center exactly 0.5, bounds pinned to a tight tolerance."""
    lo, hi = wilson_interval(50, 100)
    # Pinned from the closed-form formula (see _wilson_reference).
    assert lo == pytest.approx(0.4038315303659956, abs=1e-12)
    assert hi == pytest.approx(0.5961684696340044, abs=1e-12)
    # Symmetric about p = 0.5.
    assert (lo + hi) / 2.0 == pytest.approx(0.5, abs=1e-12)


def test_wilson_locked_values_75_of_100():
    lo, hi = wilson_interval(75, 100)
    assert lo == pytest.approx(0.656955364519384, abs=1e-12)
    assert hi == pytest.approx(0.8245478863771232, abs=1e-12)
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_matches_independent_reference():
    """Lock the engine to an independently coded closed form across many cells."""
    for s, n in [(1, 7), (13, 29), (50, 100), (123, 456), (75, 100)]:
        lo, hi = wilson_interval(s, n)
        rlo, rhi = _wilson_reference(s, n)
        assert lo == pytest.approx(rlo, abs=1e-12)
        assert hi == pytest.approx(rhi, abs=1e-12)


def test_wilson_zero_successes_lower_is_zero_upper_positive():
    """0/10: lower bound clamps to 0, upper is a real positive < 1."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775327998628892, abs=1e-12)
    assert 0.0 <= lo < hi < 1.0


def test_wilson_all_successes_upper_clamps_to_one():
    """10/10: upper bound clamps to 1.0, lower is a real positive > 0."""
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(0.7224672001371107, abs=1e-12)
    assert 0.0 < lo <= hi <= 1.0


def test_wilson_three_of_three_is_not_certainty():
    """The headline anti-lie case: 3/3 must NOT report a near-1 lower bound."""
    lo, hi = wilson_interval(3, 3)
    assert lo == pytest.approx(0.4385029682449546, abs=1e-12)
    assert hi == pytest.approx(1.0, abs=1e-12)
    # A 100% point estimate from 3 trials still admits ~44% as plausible.
    assert lo < 0.5


def test_wilson_trials_zero_returns_full_interval():
    """No data -> maximal ignorance interval, not a crash or a fake point."""
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(5, -1) == (0.0, 1.0)


def test_wilson_bounds_in_unit_interval_and_ordered_for_all_cells():
    """Structural invariant across the full grid: 0 <= lo <= hi <= 1."""
    for n in range(1, 60):
        for s in range(0, n + 1):
            lo, hi = wilson_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0, (s, n, lo, hi)


def test_wilson_wider_z_widens_interval():
    """A 99% z must produce a strictly wider band than the 95% default."""
    z99 = 2.5758293035489004
    lo95, hi95 = wilson_interval(40, 100)
    lo99, hi99 = wilson_interval(40, 100, z=z99)
    assert lo99 < lo95
    assert hi99 > hi95
    assert (hi99 - lo99) > (hi95 - lo95)


def test_wilson_more_trials_tightens_interval():
    """Same point estimate, 10x the n -> strictly narrower interval."""
    w_small = wilson_interval(6, 12)
    w_large = wilson_interval(60, 120)
    width_small = w_small[1] - w_small[0]
    width_large = w_large[1] - w_large[0]
    assert width_large < width_small


# ---------------------------------------------------------------------------
# Beta-binomial posterior -- shrinkage behavior + locked values
# ---------------------------------------------------------------------------
def test_bayes_shrinks_small_sample_toward_prior():
    """3/3 raw = 100%; posterior mean must sit close to the 0.5 prior, far from 1."""
    mean, low, high = beta_binomial_posterior(3, 3, prior_mean=0.5, prior_strength=20.0)
    # alpha = 10 + 3 = 13, beta = 10 + 0 = 10 -> mean = 13/23.
    assert mean == pytest.approx(13.0 / 23.0, abs=1e-12)
    assert mean == pytest.approx(0.5652173913043478, abs=1e-12)
    # Pulled hard toward the prior, nowhere near the raw 1.0.
    assert abs(mean - 0.5) < abs(mean - 1.0)
    assert mean < 0.6
    assert 0.0 < low < mean < high < 1.0


def test_bayes_large_sample_overwhelms_prior():
    """300/400 = 0.75 raw; with strength 20 the posterior should sit near 0.75."""
    raw = 300.0 / 400.0
    mean, low, high = beta_binomial_posterior(300, 400, prior_mean=0.5, prior_strength=20.0)
    # alpha = 310, beta = 110 -> mean = 310/420.
    assert mean == pytest.approx(310.0 / 420.0, abs=1e-12)
    assert mean == pytest.approx(0.7380952380952381, abs=1e-12)
    # Within a couple of points of the raw rate -- prior has been overwhelmed.
    assert abs(mean - raw) < 0.02
    assert low < mean < high


def test_bayes_shrinkage_is_monotone_in_sample_size():
    """As n grows with the rate fixed at 0.75, the posterior mean must move
    monotonically away from the prior (0.5) toward the raw rate."""
    means = []
    for n in (4, 20, 100, 1000):
        s = int(round(0.75 * n))
        m, _, _ = beta_binomial_posterior(s, n, prior_mean=0.5, prior_strength=20.0)
        means.append(m)
    # Strictly increasing toward 0.75 from below.
    for a, b in zip(means, means[1:]):
        assert a < b
    assert all(m < 0.75 for m in means)
    assert means[-1] == pytest.approx(0.75, abs=1e-2)


def test_bayes_no_data_returns_prior_mean():
    """0/0 with prior_mean 0.5 must return exactly the prior mean."""
    mean, low, high = beta_binomial_posterior(0, 0, prior_mean=0.5, prior_strength=20.0)
    assert mean == pytest.approx(0.5, abs=1e-12)
    assert 0.0 < low < 0.5 < high < 1.0


def test_bayes_prior_strength_controls_shrinkage():
    """A stronger prior pulls the same small sample harder toward the prior mean."""
    weak, _, _ = beta_binomial_posterior(3, 3, prior_mean=0.5, prior_strength=2.0)
    strong, _, _ = beta_binomial_posterior(3, 3, prior_mean=0.5, prior_strength=200.0)
    # Both below the raw 1.0; the strong prior is closer to 0.5.
    assert 0.5 < strong < weak < 1.0
    assert abs(strong - 0.5) < abs(weak - 0.5)


def test_bayes_extreme_prior_mean_is_clamped_not_degenerate():
    """prior_mean=1.0 is clamped to 1-1e-6; posterior stays a valid CI in (0,1)."""
    mean, low, high = beta_binomial_posterior(5, 10, prior_mean=1.0, prior_strength=20.0)
    assert 0.0 < low <= mean <= high < 1.0


def test_bayes_credible_interval_brackets_mean_and_in_unit():
    """Across a grid, the credible interval brackets the mean and lives in (0,1)."""
    for s, n in [(0, 5), (1, 1), (7, 12), (40, 50), (300, 400)]:
        mean, low, high = beta_binomial_posterior(s, n, prior_mean=0.4, prior_strength=15.0)
        assert 0.0 < low <= mean <= high < 1.0, (s, n, low, mean, high)


# ---------------------------------------------------------------------------
# estimate_proportion -- the sufficiency verdict
# ---------------------------------------------------------------------------
def test_estimate_proportion_fields_consistent():
    est = estimate_proportion(75, 100, prior_mean=0.5)
    assert est.successes == 75
    assert est.trials == 100
    assert est.point == pytest.approx(0.75, abs=1e-12)
    assert est.wilson_low == pytest.approx(0.656955364519384, abs=1e-12)
    assert est.wilson_high == pytest.approx(0.8245478863771232, abs=1e-12)
    assert est.wilson_low <= est.point <= est.wilson_high


def test_estimate_proportion_insufficient_when_thin_n():
    """3 trials: below min_trials -> not sufficient regardless of interval."""
    est = estimate_proportion(3, 3, prior_mean=0.5)
    assert est.trials < 30
    # NOTE: `sufficient` is annotated `bool` but the engine returns np.bool_
    # (the && of np.float64 comparisons), so assert by value, not identity.
    assert bool(est.sufficient) is False


def test_estimate_proportion_insufficient_when_interval_too_wide():
    """n above the floor but a 50/50 split keeps the Wilson interval wide ->
    the width gate alone must veto sufficiency."""
    # n=40, p=0.5: Wilson width is ~0.30 < 0.35, so to actually trip the width
    # gate we need a small-ish n that clears min_trials but stays wide.
    est = estimate_proportion(15, 30, prior_mean=0.5, min_trials=30, max_interval_width=0.30)
    width = est.wilson_high - est.wilson_low
    assert est.trials >= 30
    assert width > 0.30
    assert bool(est.sufficient) is False


def test_estimate_proportion_sufficient_when_n_large_and_interval_tight():
    """400 trials at 0.75: clears n floor AND tight interval -> sufficient."""
    est = estimate_proportion(300, 400, prior_mean=0.5)
    width = est.wilson_high - est.wilson_low
    assert est.trials >= 30
    assert width <= 0.35
    assert bool(est.sufficient) is True


def test_estimate_proportion_str_flags_insufficient():
    """The human-readable string must carry the INSUFFICIENT marker when thin."""
    s = str(estimate_proportion(3, 3, prior_mean=0.5))
    assert "INSUFFICIENT EVIDENCE" in s
    s2 = str(estimate_proportion(300, 400, prior_mean=0.5))
    assert "INSUFFICIENT EVIDENCE" not in s2


# ---------------------------------------------------------------------------
# bootstrap_mean_ci -- determinism + structural sanity (seeded)
# ---------------------------------------------------------------------------
def test_bootstrap_is_deterministic_with_seed():
    vals = np.array([1.0, -2.0, 3.5, 0.0, 4.0, -1.0, 2.0])
    a = bootstrap_mean_ci(vals, n_boot=2000, seed=42)
    b = bootstrap_mean_ci(vals, n_boot=2000, seed=42)
    assert a == b
    # Point estimate is the raw mean; CI brackets it.
    assert a[0] == pytest.approx(float(vals.mean()), abs=1e-12)
    assert a[1] <= a[0] <= a[2]


def test_bootstrap_handles_nan_and_singletons():
    with_nan = np.array([np.nan, 5.0, np.nan])
    mean, lo, hi = bootstrap_mean_ci(with_nan, n_boot=100)
    # Only one real value -> degenerate CI at that value.
    assert mean == pytest.approx(5.0)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)

    empty = np.array([np.nan, np.nan])
    m2, l2, h2 = bootstrap_mean_ci(empty, n_boot=100)
    assert math.isnan(m2) and math.isnan(l2) and math.isnan(h2)
