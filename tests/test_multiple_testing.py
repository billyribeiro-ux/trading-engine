"""
Adversarial tests for engine.intraday.multiple_testing.

Benjamini-Hochberg results are checked against hand-stepped step-up cutoffs, and
critically against ORIGINAL input order (an unsorted p-vector is fed in and the
boolean mapping is verified). The bootstrap p-value is deterministic (seed=17
baked into the function) so exact reproducibility is asserted, and the documented
short-circuits (observed<=0 -> 1.0, n<5 -> 1.0, floor 1/(n_boot+1)) are pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.intraday.multiple_testing import (
    _bootstrap_p_value,
    benjamini_hochberg,
    correct_scenarios,
)


# ---------------------------------------------------------------------------
# 1. benjamini_hochberg correctness on KNOWN p-vectors
# ---------------------------------------------------------------------------
def test_bh_all_large_p_values_none_significant():
    # m=4, fdr=0.10. Smallest p 0.5 vs rank-1 thresh 0.025 -> nothing passes.
    assert benjamini_hochberg([0.5, 0.6, 0.7, 0.8], fdr=0.10) == [
        False,
        False,
        False,
        False,
    ]


def test_bh_one_tiny_among_large_only_it_significant():
    # m=4, fdr=0.10. Tiny p=0.001 is rank 1: thresh = (1/4)*0.10 = 0.025.
    # 0.001 <= 0.025 -> True; all others fail their thresholds.
    out = benjamini_hochberg([0.5, 0.001, 0.6, 0.7], fdr=0.10)
    assert out == [False, True, False, False]


def test_bh_returns_results_in_original_input_order_unsorted_vector():
    # Deliberately unsorted input. p = [0.7, 0.01, 0.5, 0.02], m=4, fdr=0.10.
    # sorted ascending: 0.01(idx1), 0.02(idx3), 0.5(idx2), 0.7(idx0)
    # thresholds rank1..4: 0.025, 0.05, 0.075, 0.10
    #   0.01 <= 0.025 T ; 0.02 <= 0.05 T ; 0.5 <= 0.075 F ; 0.7 <= 0.10 F
    # k_max at sorted-pos 1 (0-indexed) -> sorted sig = [T, T, F, F]
    # mapped back to original: idx1 and idx3 are True.
    out = benjamini_hochberg([0.7, 0.01, 0.5, 0.02], fdr=0.10)
    assert out == [False, True, False, True]


def test_bh_step_up_pulls_in_a_higher_p_below_its_own_threshold():
    # The defining property of step-up: a p that fails its OWN threshold is still
    # significant if a higher-ranked p passes. p=[0.04, 0.005, 0.5], m=3, fdr=0.10.
    # thresholds rank1..3: 0.0333..., 0.0667..., 0.10
    # sorted: 0.005(idx1), 0.04(idx0), 0.5(idx2)
    #   0.005 <= 0.0333 T ; 0.04 <= 0.0667 T ; 0.5 <= 0.10 F -> k_max=1
    # sorted sig = [T, T, F] -> idx0 and idx1 True, idx2 False.
    # Note: 0.04 alone vs its rank-1 thresh 0.0333 would FAIL; step-up rescues it.
    out = benjamini_hochberg([0.04, 0.005, 0.5], fdr=0.10)
    assert out == [True, True, False]


def test_bh_monotone_block_all_pass():
    # p=[0.01,0.02,0.03,0.04,0.05], m=5, fdr=0.10.
    # thresholds: 0.02,0.04,0.06,0.08,0.10 ; every p <= its threshold -> all True.
    out = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], fdr=0.10)
    assert out == [True, True, True, True, True]


# ---------------------------------------------------------------------------
# 2. benjamini_hochberg edge cases
# ---------------------------------------------------------------------------
def test_bh_empty_returns_empty_list():
    assert benjamini_hochberg([], fdr=0.10) == []


def test_bh_single_p_below_threshold():
    # m=1, thresh = (1/1)*0.10 = 0.10. 0.04 <= 0.10 -> True.
    assert benjamini_hochberg([0.04], fdr=0.10) == [True]


def test_bh_single_p_above_threshold():
    # 0.50 > 0.10 -> False.
    assert benjamini_hochberg([0.50], fdr=0.10) == [False]


# ---------------------------------------------------------------------------
# 3. _bootstrap_p_value -- determinism + documented short-circuits
# ---------------------------------------------------------------------------
def test_bootstrap_clearly_positive_sample_hits_the_floor_and_is_reproducible():
    rng = np.random.default_rng(123)
    pos = 1.0 + rng.normal(0, 0.01, size=200)  # mean ~1, tiny noise
    p = _bootstrap_p_value(pos)
    # No null resample (centered to mean 0) reaches the observed ~1.0 mean, so the
    # finite-resampling floor 1/(n_boot+1) = 1/10001 applies.
    assert p == pytest.approx(1.0 / 10001)
    # Internal seed=17 -> deterministic across calls.
    assert _bootstrap_p_value(pos) == p


def test_bootstrap_nontrivial_positive_sample_is_deterministic():
    small = np.array([0.1, -0.05, 0.2, 0.05, -0.1, 0.15, 0.0, 0.08])
    # mean = 0.05375 > 0, modest -> p strictly between floor and 1.
    p1 = _bootstrap_p_value(small)
    p2 = _bootstrap_p_value(small)
    assert p1 == p2 == pytest.approx(0.052)
    assert (1.0 / 10001) < p1 < 1.0


def test_bootstrap_negative_mean_short_circuits_to_one():
    rng = np.random.default_rng(123)
    neg = -1.0 + rng.normal(0, 0.01, size=200)
    assert _bootstrap_p_value(neg) == 1.0


def test_bootstrap_exactly_zero_mean_short_circuits_to_one():
    zero = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # mean exactly 0
    assert zero.mean() == 0.0
    assert _bootstrap_p_value(zero) == 1.0  # observed <= 0 branch


def test_bootstrap_small_n_returns_one_even_if_hugely_positive():
    # n=4 < 5 -> 1.0 regardless of how positive the sample is.
    assert _bootstrap_p_value(np.array([100.0, 100.0, 100.0, 100.0])) == 1.0


def test_bootstrap_drops_nans_then_applies_small_n_rule():
    # 5 entries, 2 NaN -> 3 finite -> n<5 -> 1.0
    arr = np.array([1.0, np.nan, 1.0, np.nan, 1.0])
    assert _bootstrap_p_value(arr) == 1.0


# ---------------------------------------------------------------------------
# 4. correct_scenarios -- one profitable scenario among nulls
# ---------------------------------------------------------------------------
def _build_scenarios():
    rng = np.random.default_rng(42)
    return {
        # Clearly profitable: mean ~0.5, tiny noise -> p hits the floor.
        ("winner", "long"): 0.5 + rng.normal(0, 0.1, size=100),
        # Clearly negative -> observed<=0 -> p = 1.0 (deterministic null).
        ("loser", "short"): -0.5 + rng.normal(0, 0.1, size=100),
        # Mildly negative -> also p = 1.0.
        ("flat", "long"): -0.2 + rng.normal(0, 0.1, size=100),
    }


def test_correct_scenarios_winner_is_smallest_p_and_significant():
    data = _build_scenarios()
    m = len(data)
    res = correct_scenarios(data, fdr=0.10, alpha=0.05)

    # Sorted by p ascending.
    ps = [t.p_value for t in res]
    assert ps == sorted(ps)

    # Ranks are a permutation of 1..m, contiguous.
    assert sorted(t.rank for t in res) == list(range(1, m + 1))

    by_key = {(t.scenario, t.side): t for t in res}
    winner = by_key[("winner", "long")]
    loser = by_key[("loser", "short")]
    flat = by_key[("flat", "long")]

    # Winner is rank 1 with the smallest p (the floor), and survives BH.
    assert winner.rank == 1
    assert winner.p_value == pytest.approx(1.0 / 10001)
    assert winner.p_value == min(ps)
    assert winner.bh_significant is True

    # The nulls have p = 1.0 and are NOT significant by BH.
    assert loser.p_value == 1.0
    assert flat.p_value == 1.0
    assert loser.bh_significant is False
    assert flat.bh_significant is False

    # Means line up with the planted truth.
    assert winner.mean_net_r > 0.4
    assert loser.mean_net_r < 0.0
    assert flat.mean_net_r < 0.0


def test_correct_scenarios_bonferroni_uses_alpha_over_m():
    data = _build_scenarios()
    m = len(data)  # 3
    alpha = 0.05
    res = correct_scenarios(data, fdr=0.10, alpha=alpha)
    by_key = {(t.scenario, t.side): t for t in res}

    bonf_alpha = alpha / m  # 0.016666...
    # Winner's floor p (~9.999e-05) is below alpha/m -> Bonferroni significant.
    winner = by_key[("winner", "long")]
    assert winner.p_value <= bonf_alpha
    assert winner.bonferroni_significant is True

    # Each flag must equal (p <= alpha/m), exactly.
    for t in res:
        assert t.bonferroni_significant == (t.p_value <= bonf_alpha)

    # Nulls at p=1.0 are far above alpha/m -> not Bonferroni significant.
    assert by_key[("loser", "short")].bonferroni_significant is False
    assert by_key[("flat", "long")].bonferroni_significant is False


def test_correct_scenarios_bh_threshold_matches_rank_over_m_times_fdr():
    data = _build_scenarios()
    m = len(data)
    fdr = 0.10
    res = correct_scenarios(data, fdr=fdr, alpha=0.05)
    for t in res:
        assert t.bh_threshold == pytest.approx((t.rank / m) * fdr)
