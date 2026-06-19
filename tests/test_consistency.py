"""
Re-lock for historical bug #3: STRUCTURE vs LEG ROLES disagreed (11 legs vs 5).

The invariant under test (from engine/session/runner.py:dissect_real_session):

    dec     = decompose(session)
    dis     = dissect_session(session, decomposition=dec)
    nested  = build_nested_structure(dec, dis.scale_atr)   # NaN scale -> []

`dissect_session` builds its classified legs via
`build_skeleton(decomposition, major_scale=dis.scale_atr, ...)`, and
`build_nested_structure(dec, dis.scale_atr)` builds its STRUCTURE major legs
via `build_skeleton(decomposition, dis.scale_atr)` -- the SAME major_scale, the
SAME `merge_insignificant_swings` + zero-duration drop. So the STRUCTURE major
legs MUST be identical, leg-for-leg, to the legs under `classified_legs`. If they
ever diverge again (different counts, different boundaries), bug #3 is back.

These tests are deterministic: every session is built from a seeded
`np.random.default_rng(SEED)` via the shared synthetic helpers.
"""

from __future__ import annotations

import _synth as S
import numpy as np
import pytest

from engine.session.dissect import dissect_session
from engine.session.nested import build_nested_structure
from engine.session.pivots import decompose

SEEDS = tuple(range(20))


def _dissect_like_runner(session):
    """Replicate EXACTLY engine/session/runner.py:dissect_real_session's pipeline.

    Returns (decomposition, dissection, nested). The NaN guard mirrors the runner
    (`dis.scale_atr == dis.scale_atr` is False only when scale_atr is NaN).
    """
    dec = decompose(session)
    dis = dissect_session(session, decomposition=dec)
    nested = (
        build_nested_structure(dec, dis.scale_atr)
        if dis.scale_atr == dis.scale_atr  # NaN guard, byte-for-byte with runner
        else []
    )
    return dec, dis, nested


def _leg_key(leg):
    """The identity tuple bug #3 demands match: span + price endpoints."""
    return (leg.start_index, leg.end_index, leg.start_price, leg.end_price)


@pytest.mark.parametrize("seed", SEEDS)
def test_structure_majors_identical_to_classified_legs(seed):
    """Bug-#3 re-lock: STRUCTURE majors == LEG-ROLES legs, leg-for-leg.

    Across each of 20 seeded multi-leg sessions, the (start_index, end_index,
    start_price, end_price) tuples of [nl.major for nl in nested] must be
    IDENTICAL -- same length, same order, same values -- to those of
    [cl.leg for cl in dis.classified_legs]. A count mismatch OR a boundary
    mismatch is the historical bug.
    """
    rng = np.random.default_rng(seed)
    session = S.multileg_session(rng)
    _dec, dis, nested = _dissect_like_runner(session)

    structure_keys = [_leg_key(nl.major) for nl in nested]
    classified_keys = [_leg_key(cl.leg) for cl in dis.classified_legs]

    # Same count (the literal "11 legs vs 5" symptom of bug #3).
    assert len(structure_keys) == len(classified_keys), (
        f"seed={seed}: STRUCTURE has {len(structure_keys)} majors but "
        f"LEG ROLES has {len(classified_keys)} legs -- bug #3 regression.\n"
        f"  structure={structure_keys}\n  classified={classified_keys}"
    )
    # Identical legs, in order (boundaries + prices, not just count).
    assert structure_keys == classified_keys, (
        f"seed={seed}: STRUCTURE majors diverge from classified legs.\n"
        f"  structure={structure_keys}\n  classified={classified_keys}"
    )


def test_suite_exercises_real_structure():
    """The re-lock must run against REAL structure, not empty dissections.

    If every synthetic session produced 0 legs, the parametrized test above would
    pass vacuously ([] == []). Assert that the seed set actually drives the
    invariant: at least some sessions must produce >=4 legs, and none of them may
    be empty (an empty dissection here would mean the synthetic generator stopped
    producing structure and the re-lock has gone toothless).
    """
    leg_counts = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        session = S.multileg_session(rng)
        _dec, dis, _nested = _dissect_like_runner(session)
        leg_counts.append(len(dis.classified_legs))

    n_with_real_structure = sum(1 for c in leg_counts if c >= 4)
    assert n_with_real_structure >= 1, (
        f"No seeded session produced >=4 legs; the bug-#3 re-lock is vacuous. "
        f"leg_counts={leg_counts}"
    )
    # Stronger: none should be empty -- a 0-leg session would silently neuter the
    # parametrized identity check ([] == []).
    assert all(c > 0 for c in leg_counts), (
        f"Some seeded session produced an empty dissection -- the re-lock would "
        f"pass vacuously for it. leg_counts={leg_counts}"
    )
