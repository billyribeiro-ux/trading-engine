"""
Adversarial tests for engine/session/dissect.py.

These tests try to BREAK the dissection layer, not to confirm "it ran":
  * leg roles must be a closed subset of LegRole and the summary count must
    agree with the actual leg list (no silent off-by-one);
  * group_vwap_events must collapse consecutive same-type runs into ONE group
    with the strongest member as representative, and must NOT merge across a
    type change or a too-large gap;
  * level detection must classify prior-day H/L/close with the correct
    LevelKind and held/broke outcome, and report each level at most once;
  * a degenerate (structureless) session must return an empty, NaN-scale
    dissection WITHOUT raising, and narrate() must still work;
  * narrate() must produce a multi-line string referencing symbol + date.

Determinism: every synthetic session is built from np.random.default_rng(SEED)
or from a hand-specified close path with hand-computable level interactions.
"""

from __future__ import annotations

import _synth as S
import numpy as np
import pandas as pd

from engine.session.dissect import (
    LegRole,
    LevelKind,
    VwapEvent,
    VwapEventType,
    dissect_session,
    group_vwap_events,
)
from engine.session.session import PriorDay

SEED = 0


# --------------------------------------------------------------------------- #
# 1. Leg classification: roles are a closed set; summary count agrees.        #
# --------------------------------------------------------------------------- #
def test_classified_leg_roles_are_subset_and_count_matches():
    """Every leg's roles are valid LegRole members and summary['n_legs'] agrees.

    Adversarial intent: catch (a) a stray non-enum role leaking in, (b) an empty
    role tuple (the classifier guarantees at least UNCLASSIFIED), and (c) a
    summary count that drifts from the actual leg list -- the exact shape of the
    "STRUCTURE vs LEG ROLES disagreed" family of bugs.
    """
    rng = np.random.default_rng(SEED)
    session = S.multileg_session(rng)
    dis = dissect_session(session)

    # Real structure must exist or this test is vacuous.
    assert len(dis.classified_legs) >= 4, (
        f"expected real structure, got {len(dis.classified_legs)} legs"
    )

    valid_roles = set(LegRole)
    for cl in dis.classified_legs:
        assert isinstance(cl.roles, tuple)
        assert len(cl.roles) >= 1, "a leg must carry at least one role"
        assert set(cl.roles) <= valid_roles, f"unknown role in {cl.roles}"
        # roles are deduped (dict.fromkeys) -- no duplicate role on one leg.
        assert len(cl.roles) == len(set(cl.roles)), f"duplicate role in {cl.roles}"

    assert dis.summary["n_legs"] == len(dis.classified_legs), (
        f"summary n_legs={dis.summary['n_legs']} != {len(dis.classified_legs)} actual legs"
    )


# --------------------------------------------------------------------------- #
# 2. group_vwap_events: collapse runs, keep strongest, split on type/gap.      #
# --------------------------------------------------------------------------- #
def _mk_event(etype: VwapEventType, idx: int, outcome: float) -> VwapEvent:
    t0 = pd.Timestamp("2026-06-01 09:30")
    return VwapEvent(
        type=etype,
        index=idx,
        time=t0 + pd.Timedelta(minutes=idx),
        price=100.0,
        vwap=99.0,
        outcome_atr=outcome,
        horizon_bars=10,
    )


def test_group_vwap_events_collapses_run_and_splits_on_type_and_gap():
    """A run of >=3 same-type events within max_gap_bars collapses into ONE group.

    Construction (max_gap_bars=4):
      idx 5,7,9  RETEST_FAIL  (gaps 2,2 -> one run of 3)
      idx 11     RECLAIM      (type change -> new group)
      idx 30     RECLAIM      (gap 19 > 4 -> new group despite same type)

    Asserts: 3 groups; the first has count==3 and representative == the member
    with the LARGEST |outcome_atr| (the -1.3 one, not the +0.5 first or +0.2
    last); the two RECLAIMs do NOT merge.
    """
    max_gap = 4
    run_strongest = -1.3  # largest magnitude in the RETEST_FAIL run
    events = [
        _mk_event(VwapEventType.RETEST_FAIL, 5, 0.5),
        _mk_event(VwapEventType.RETEST_FAIL, 7, run_strongest),
        _mk_event(VwapEventType.RETEST_FAIL, 9, 0.2),
        _mk_event(VwapEventType.RECLAIM, 11, 2.0),  # type change splits the run
        _mk_event(VwapEventType.RECLAIM, 30, 0.1),  # gap 19 > 4 splits again
    ]

    groups = group_vwap_events(events, max_gap_bars=max_gap)

    assert len(groups) == 3, f"expected 3 groups, got {len(groups)}: {groups}"

    g0 = groups[0]
    assert g0.type is VwapEventType.RETEST_FAIL
    assert g0.count == 3, f"run of 3 collapsed to count={g0.count}"
    assert g0.representative.outcome_atr == run_strongest, (
        "representative must be the largest |outcome_atr| member, got "
        f"{g0.representative.outcome_atr}"
    )
    assert g0.representative.index == 7
    assert g0.first_time == events[0].time
    assert g0.last_time == events[2].time

    # The two RECLAIMs stayed separate (gap > max_gap), each count==1.
    g1, g2 = groups[1], groups[2]
    assert g1.type is VwapEventType.RECLAIM and g1.count == 1
    assert g2.type is VwapEventType.RECLAIM and g2.count == 1
    assert g1.representative.index == 11
    assert g2.representative.index == 30


def test_group_vwap_events_boundary_gap_equal_max_is_grouped():
    """A gap EXACTLY equal to max_gap_bars groups (the comparison is <=)."""
    events = [
        _mk_event(VwapEventType.LOSS, 0, -0.5),
        _mk_event(VwapEventType.LOSS, 4, -0.9),  # gap == max_gap_bars -> same run
    ]
    groups = group_vwap_events(events, max_gap_bars=4)
    assert len(groups) == 1
    assert groups[0].count == 2
    # Strongest is -0.9.
    assert groups[0].representative.outcome_atr == -0.9


def test_group_vwap_events_empty_returns_empty():
    """No events -> no groups (guard clause must not raise)."""
    assert group_vwap_events([], max_gap_bars=4) == []


# --------------------------------------------------------------------------- #
# 3. Level events: prior-day H/L/close detection, held/broke, once-only.       #
# --------------------------------------------------------------------------- #
def test_level_events_prior_day_held_and_broke_detected_once():
    """Prior-day levels inside the path are detected with correct kind+outcome.

    Hand-built path (wick 0.02, so tol = 0.15*atr is small):
      * rise 100 -> 105 then HOVER at ~105 for >=10 bars
            -> PRIOR_HIGH (105.0) is TAGGED and HELD (price stays at the level
               through the horizon, so neither broke_up nor broke_down).
      * then fall through 99 and stay below
            -> PRIOR_LOW (99.0) is TAGGED and BROKE (closes well below within
               the horizon).
    Asserts the correct LevelKind for each, the held/broke booleans, that
    held == not broke, and that EACH level appears at most once (the engine
    reports only the first test of each level).
    """
    up = list(np.linspace(100.0, 105.0, 8))
    hover = [105.0, 104.9, 105.1, 105.0, 104.95, 105.05, 105.0, 104.9, 105.1, 105.0, 105.0]
    drop = list(np.linspace(105.0, 95.0, 14))
    closes = up + hover + drop

    prior = PriorDay(high=105.0, low=99.0, close=102.0)
    session = S.session_from_closes(closes, prior_day=prior, wick=0.02)
    dis = dissect_session(session)

    by_kind = {}
    for le in dis.level_events:
        # held and broke are mutually exclusive complements by construction.
        assert le.held == (not le.broke), (
            f"{le.kind} held={le.held} broke={le.broke} not complementary"
        )
        by_kind.setdefault(le.kind, []).append(le)

    # Each level reported at most once (first test only).
    for kind, evs in by_kind.items():
        assert len(evs) == 1, f"{kind} reported {len(evs)} times, expected 1"

    assert LevelKind.PRIOR_HIGH in by_kind, "prior_high was never tested"
    assert LevelKind.PRIOR_LOW in by_kind, "prior_low was never tested"

    ph = by_kind[LevelKind.PRIOR_HIGH][0]
    assert ph.level == 105.0
    assert ph.held is True and ph.broke is False, (
        f"prior_high should HOLD (price hovers at the level), got held={ph.held} broke={ph.broke}"
    )

    pl = by_kind[LevelKind.PRIOR_LOW][0]
    assert pl.level == 99.0
    assert pl.broke is True and pl.held is False, (
        f"prior_low should BREAK (price closes well below), got held={pl.held} broke={pl.broke}"
    )
    # Break direction sanity: a downside break leaves a negative outcome_atr.
    assert pl.outcome_atr < 0, f"downside break should be negative, got {pl.outcome_atr}"


def test_level_events_absent_when_no_prior_day():
    """With no PriorDay, no PRIOR_* level events can be emitted."""
    rng = np.random.default_rng(SEED)
    session = S.multileg_session(rng)  # multileg_session attaches no prior_day
    assert session.prior_day is None
    dis = dissect_session(session)
    prior_kinds = {
        LevelKind.PRIOR_HIGH,
        LevelKind.PRIOR_LOW,
        LevelKind.PRIOR_CLOSE,
    }
    assert not any(le.kind in prior_kinds for le in dis.level_events), (
        "prior-level events emitted despite no prior_day"
    )


# --------------------------------------------------------------------------- #
# 4. Degenerate session: empty / NaN-scale dissection, no exception.           #
# --------------------------------------------------------------------------- #
def test_degenerate_session_returns_empty_nan_dissection_without_raising():
    """A structureless session yields an empty NaN-scale dissection, no raise.

    NOTE on the synthetic generator: the task hint suggested constant_tr_session,
    but that generator's per-bar wick oscillation actually DOES produce ~n legs at
    fine scales (its mean TR == tr > 0, so the 0.25-ATR threshold confirms swings
    every bar). The genuinely degenerate case is a perfectly FLAT, ZERO-WICK
    session: every bar has high==low==close==open, so mean true range is 0, the
    atr_mean floor (close-to-close std) is also 0, and atr_mean is NaN. With a
    non-finite ATR, pivots_at_scale returns [] at every scale, primary_scale
    returns None, and dissect_session takes its "no structural pivots" branch.
    """
    session = S.session_from_closes([100.0] * 40, wick=0.0)
    # Precondition: ATR really is non-finite (drives the degenerate branch).
    assert not np.isfinite(session.atr_mean), (
        f"expected NaN atr_mean for the degenerate session, got {session.atr_mean}"
    )

    dis = dissect_session(session)  # must NOT raise

    assert dis.classified_legs == ()
    assert dis.vwap_events == ()
    assert dis.level_events == ()
    assert np.isnan(dis.scale_atr), f"expected NaN scale_atr, got {dis.scale_atr}"
    assert dis.summary.get("note") == "no structural pivots found"

    # narrate() must still work on a NaN-scale dissection (string formatting of
    # NaN, no legs to iterate).
    text = dis.narrate()
    assert isinstance(text, str) and text.strip()


# --------------------------------------------------------------------------- #
# 5. narrate(): non-empty, multi-line, references symbol + date.               #
# --------------------------------------------------------------------------- #
def test_narrate_is_multiline_and_references_symbol_and_date():
    """narrate() returns the chart read back in words -- referencing who/when."""
    rng = np.random.default_rng(SEED)
    symbol = "ADV"
    session = S.multileg_session(rng, symbol=symbol)
    dis = dissect_session(session)

    text = dis.narrate()
    assert isinstance(text, str)
    lines = text.splitlines()
    assert len(lines) >= 2, f"expected multi-line narration, got {lines!r}"
    assert symbol in text, "narration must reference the symbol"
    assert str(session.date.date()) in text, "narration must reference the date"
    # Real structure -> one narration line per classified leg plus a header and
    # the VWAP-events summary line.
    assert len(lines) == len(dis.classified_legs) + 2, (
        f"narration line count {len(lines)} != legs "
        f"{len(dis.classified_legs)} + header + vwap-summary"
    )
