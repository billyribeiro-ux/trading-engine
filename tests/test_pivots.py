"""
Adversarial tests for engine.session.pivots.

These tests plant zigzags with a KNOWN number of swings, then assert the
detected pivot count, the terminal-pivot closing behavior, leg geometry, and --
the decisive one -- that merge_insignificant_swings absorbs ONLY shallow AND
brief counter-swings (guarding the documented "78% pullback wrongly absorbed"
bug). primary_scale/build_skeleton are checked for structural soundness.

Randomness is seeded via np.random.default_rng(SEED). Leg fixtures are built by
hand so every assertion compares against an independently derived expectation.
"""

from __future__ import annotations

import _synth as S
import numpy as np
import pandas as pd

from engine.session.pivots import (
    Leg,
    PivotType,
    build_skeleton,
    decompose,
    legs_from_pivots,
    merge_insignificant_swings,
    pivots_at_scale,
)

SEED = 20260618


def _planted_zigzag_session():
    """A clean 4-pivot zigzag whose final move (up to 130) does NOT reverse, so
    a terminal pivot should close it at the last bar. Returns (session, scale)
    where `scale` (in ATR units) yields a threshold of ~5 price points -- well
    below each ~10-point swing, well above the ~0.1 wick noise."""
    closes = []
    for x in np.linspace(100, 110, 11):
        closes.append(float(x))
    for x in np.linspace(110, 100, 11)[1:]:
        closes.append(float(x))
    for x in np.linspace(100, 120, 21)[1:]:
        closes.append(float(x))
    for x in np.linspace(120, 130, 11)[1:]:
        closes.append(float(x))
    s = S.session_from_closes(closes)
    scale = 5.0 / s.atr_mean  # threshold ~= 5 price points
    return s, scale


def _mk_leg(s_idx, e_idx, s_px, e_px, scale=1.0):
    t0 = pd.Timestamp("2026-06-01 09:30")
    return Leg(
        start_index=s_idx,
        end_index=e_idx,
        start_time=t0 + pd.Timedelta(minutes=s_idx),
        end_time=t0 + pd.Timedelta(minutes=e_idx),
        start_price=float(s_px),
        end_price=float(e_px),
        scale_atr=scale,
    )


# ----------------------------------------------------------------------------
# 1. pivots_at_scale on a planted zigzag + terminal pivot closes final swing
# ----------------------------------------------------------------------------
def test_pivots_at_scale_finds_planted_swings_and_terminal_pivot():
    s, scale = _planted_zigzag_session()
    piv = pivots_at_scale(s, scale)

    # 3 confirmed swings (LOW, HIGH, LOW) + 1 terminal HIGH = 4 pivots.
    assert len(piv) == 4
    assert [p.type for p in piv] == [
        PivotType.LOW,
        PivotType.HIGH,
        PivotType.LOW,
        PivotType.HIGH,
    ]
    # The interior pivots land at the planted extremes (~100, ~110, ~100).
    assert piv[0].price < 101  # the initial low near 100
    assert piv[1].price > 109  # the high near 110
    assert piv[2].price < 101  # the low near 100

    # The closing move did NOT reverse, so the terminal pivot reaches the final
    # bar and is confirmed there (no lookahead -- confirmed_index == last bar).
    last = len(s) - 1
    assert piv[-1].index == last
    assert piv[-1].confirmed_index == last
    assert piv[-1].price > 129  # the unreversed top near 130


def test_pivots_confirmation_is_causal():
    """Every confirmed (non-terminal) pivot's confirmation bar is at or after
    its extreme bar -- no pivot is 'known' before price proves it."""
    s, scale = _planted_zigzag_session()
    piv = pivots_at_scale(s, scale)
    for p in piv:
        assert p.confirmed_index >= p.index


def test_pivots_at_scale_empty_when_threshold_exceeds_all_swings():
    """A scale so coarse that no swing ever retraces it yields no pivots (the
    directional-change algorithm never confirms; no terminal pivot is forced)."""
    s, _ = _planted_zigzag_session()
    huge_scale = 1000.0 / s.atr_mean  # threshold ~1000 points, far above any swing
    assert pivots_at_scale(s, huge_scale) == []


# ----------------------------------------------------------------------------
# 2. legs_from_pivots geometry
# ----------------------------------------------------------------------------
def test_legs_connect_consecutive_pivots_with_correct_geometry():
    s, scale = _planted_zigzag_session()
    piv = pivots_at_scale(s, scale)
    legs = legs_from_pivots(s, piv)

    assert len(legs) == len(piv) - 1
    for a, c, lg in zip(piv[:-1], piv[1:], legs):
        # endpoints copied exactly from the pivots
        assert lg.start_index == a.index and lg.end_index == c.index
        assert lg.start_price == a.price and lg.end_price == c.price
        # direction/magnitude/bars derived correctly
        assert lg.direction == ("up" if c.price > a.price else "down")
        assert lg.magnitude == abs(c.price - a.price)
        assert lg.bars == c.index - a.index
    # The planted shape alternates up/down/up.
    assert [lg.direction for lg in legs] == ["up", "down", "up"]


def test_legs_from_pivots_needs_two_pivots():
    s, _ = _planted_zigzag_session()
    assert legs_from_pivots(s, []) == []
    one = pivots_at_scale(s, 5.0 / s.atr_mean)[:1]
    assert legs_from_pivots(s, one) == []


# ----------------------------------------------------------------------------
# 3. merge_insignificant_swings -- the decisive 78%-pullback guard
# ----------------------------------------------------------------------------
def test_merge_absorbs_shallow_and_brief_counter_swing():
    """a(up) / b(small,2-bar down) / c(up to new high): b is shallow (10% of the
    combined move) AND brief (<=3 bars) AND the move continues to a new extreme,
    so the three legs collapse into one."""
    a = _mk_leg(0, 10, 100, 110)
    b = _mk_leg(10, 12, 110, 108)  # mag 2, 2 bars
    c = _mk_leg(12, 22, 108, 120)  # new high 120 > 110
    combined = abs(120 - 100)
    assert b.magnitude / combined < 0.40  # shallow
    assert b.bars <= 3  # brief
    out = merge_insignificant_swings([a, b, c])
    assert len(out) == 1
    merged = out[0]
    assert merged.start_index == 0 and merged.end_index == 22
    assert merged.start_price == 100 and merged.end_price == 120
    assert merged.direction == "up"


def test_merge_does_NOT_absorb_deep_counter_swing():
    """The 78%-retrace guard: a deep counter-swing (>40% of the combined move)
    is structural and must NOT be absorbed even though it is brief."""
    a = _mk_leg(0, 10, 100, 110)
    b = _mk_leg(10, 12, 110, 102.2)  # retraces 7.8 of the 10-pt advance, 2 bars
    c = _mk_leg(12, 22, 102.2, 112)
    combined = abs(112 - 100)  # 12
    counter_frac = b.magnitude / combined  # 7.8/12 = 0.65 > 0.40
    assert counter_frac > 0.40
    assert b.bars <= 3  # brief, yet must survive because it is deep
    out = merge_insignificant_swings([a, b, c])
    assert len(out) == 3  # untouched
    assert [lg.direction for lg in out] == ["up", "down", "up"]


def test_merge_does_NOT_absorb_long_counter_swing():
    """A shallow but LONG (>3 bar) counter-swing is structural duration-wise and
    must NOT be absorbed."""
    a = _mk_leg(0, 10, 100, 110)
    b = _mk_leg(10, 15, 110, 108)  # shallow (mag 2) but 5 bars
    c = _mk_leg(15, 25, 108, 120)
    combined = abs(120 - 100)
    assert b.magnitude / combined < 0.40  # shallow
    assert b.bars > 3  # long -> must survive
    out = merge_insignificant_swings([a, b, c])
    assert len(out) == 3


def test_merge_does_NOT_absorb_when_no_new_extreme():
    """If the continuation fails to exceed a's end (no new extreme), the
    counter-swing is structural (a failed breakout) and is not absorbed."""
    a = _mk_leg(0, 10, 100, 110)
    b = _mk_leg(10, 12, 110, 108)  # shallow, brief
    c = _mk_leg(12, 22, 108, 109)  # fails to exceed 110 -> no new extreme
    out = merge_insignificant_swings([a, b, c])
    assert len(out) == 3


def test_merge_is_identity_on_too_few_legs():
    a = _mk_leg(0, 10, 100, 110)
    b = _mk_leg(10, 20, 110, 100)
    assert merge_insignificant_swings([a, b]) == [a, b]


# ----------------------------------------------------------------------------
# 4. build_skeleton drops zero-duration legs, ignores refine_scale
# ----------------------------------------------------------------------------
def test_build_skeleton_equals_merge_minus_zero_duration_legs():
    rng = np.random.default_rng(SEED)
    s = S.multileg_session(rng)
    dec = decompose(s)
    ps = dec.primary_scale(n_bars=len(s))
    assert ps is not None

    merged = merge_insignificant_swings(dec.legs_by_scale.get(ps, []))
    expected = [lg for lg in merged if lg.end_index > lg.start_index]
    skel = build_skeleton(dec, ps, n_bars=len(s))
    assert skel == expected
    # No zero-duration legs survive in the skeleton.
    assert all(lg.end_index > lg.start_index for lg in skel)


def test_build_skeleton_actually_drops_a_terminal_zero_duration_leg():
    """Construct legs where a terminal blip lands on the final bar (zero
    duration). build_skeleton must drop it; merge alone would keep it."""
    # Three real legs that do NOT merge (deep counters), plus a zero-dur tail.
    legs = [
        _mk_leg(0, 10, 100, 110),
        _mk_leg(10, 20, 110, 100),
        _mk_leg(20, 30, 100, 112),
        _mk_leg(30, 30, 112, 112),  # zero-duration terminal blip
    ]
    from engine.session.pivots import MultiScaleDecomposition

    dec = MultiScaleDecomposition(
        session_date=pd.Timestamp("2026-06-01"),
        symbol="X",
        scales=(1.0,),
        pivots_by_scale={1.0: []},
        legs_by_scale={1.0: legs},
        atr_mean=1.0,
    )
    skel = build_skeleton(dec, 1.0)
    assert len(skel) == 3
    assert all(lg.end_index > lg.start_index for lg in skel)


def test_build_skeleton_ignores_refine_scale_and_n_bars():
    """refine_scale / n_bars are accepted for signature stability but must not
    change the output (the documented re-fragmentation bug was removed)."""
    rng = np.random.default_rng(SEED)
    s = S.multileg_session(rng)
    dec = decompose(s)
    ps = dec.primary_scale(n_bars=len(s))
    base = build_skeleton(dec, ps)
    with_refine = build_skeleton(dec, ps, refine_scale=0.5, n_bars=len(s))
    with_other_refine = build_skeleton(dec, ps, refine_scale=3.0, n_bars=len(s))
    assert base == with_refine == with_other_refine


# ----------------------------------------------------------------------------
# 5. primary_scale: bounded leg count + real structure on a multileg session
# ----------------------------------------------------------------------------
def test_primary_scale_merged_count_within_bounds():
    rng = np.random.default_rng(SEED)
    s = S.multileg_session(rng)
    dec = decompose(s)
    min_legs, max_legs = 4, 14
    ps = dec.primary_scale(n_bars=len(s), min_legs=min_legs, max_legs=max_legs)
    assert ps is not None
    merged = merge_insignificant_swings(dec.legs_by_scale.get(ps, []))
    assert min_legs <= len(merged) <= max_legs


def test_primary_scale_finds_real_structure_multiple_seeds():
    """On several seeded multileg sessions, primary_scale must surface >=4 legs
    of genuine structure -- it cannot collapse a clean 6-leg path to a line."""
    for seed in (0, 1, SEED, 99):
        rng = np.random.default_rng(seed)
        s = S.multileg_session(rng)
        dec = decompose(s)
        ps = dec.primary_scale(n_bars=len(s))
        assert ps is not None, f"no primary scale for seed {seed}"
        skel = build_skeleton(dec, ps, n_bars=len(s))
        assert len(skel) >= 4, f"seed {seed}: only {len(skel)} legs"
        # Real structure alternates direction (no two same-direction legs in a
        # row after merging).
        dirs = [lg.direction for lg in skel]
        assert all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1))
