"""
Adversarial tests for engine/intraday/walkforward.py.

The Prime Directive (from _synth.py): tests must actively try to BREAK the
engine. Here we attack the SignalOutcome-based purged/embargoed walk-forward.

The central claim of walk_forward is "no lookahead": an in-sample (IS) window
used to estimate an edge must NOT contain any session that belongs to the
out-of-sample (OOS) fold it is being measured against, a PURGE gap of
`purge_sessions` must separate the IS tail from the OOS start, and an EMBARGO of
`embargo_sessions` must separate a finished OOS fold from the next IS window.

We construct SignalOutcomes with KNOWN session dates and a deterministic,
session-separable net_r so we can assert EXACTLY which sessions land in IS vs
OOS vs purged vs embargoed, by reconstructing the engine's own fold boundaries
and (separately) by planting a leak that WOULD change the pooled result if the
purge were not honoured.

Public API exercised:
    walk_forward(outcomes, cfg) -> dict[(scenario, side), WalkForwardResult]
    WalkForwardConfig(n_folds, purge_sessions, embargo_sessions,
                      min_is_sessions, min_oos_events)
    WalkForwardResult(.folds, .pooled_is_net_r, .pooled_oos_net_r, .decay_r,
                      .n_oos_events, .trustworthy, ...)
    FoldResult(.fold, .is_sessions, .oos_sessions, .is_events, .oos_events, ...)
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.intraday.backtest import SignalOutcome
from engine.intraday.reversal import ReversalSide, ReversalSignal
from engine.intraday.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward,
)

SCENARIO = "scenario_x"
BASE_DATE = pd.Timestamp("2026-01-05")  # a Monday; arbitrary but fixed


def _session_for(day_index: int) -> pd.Timestamp:
    """Map an ordinal session index to a calendar day (1 session == 1 day)."""
    return BASE_DATE + pd.Timedelta(days=day_index)


def _make_outcome(
    session: pd.Timestamp,
    net_r: float,
    *,
    side: ReversalSide = ReversalSide.BULLISH,
    scenario: str = SCENARIO,
    retest: bool = True,
) -> SignalOutcome:
    """A SignalOutcome carrying a KNOWN session and net_r; other fields are inert
    for walk_forward (it reads only signal.session, signal.scenario,
    signal.side.value, and net_r)."""
    sig = ReversalSignal(
        symbol="TEST",
        session=session,
        side=side,
        scenario=scenario,
        signal_time=session + pd.Timedelta(hours=9, minutes=31),
        signal_index=1,
        entry_price=100.0,
        origin_extreme=101.0,
        counter_extreme=99.0,
        vwap_at_signal=100.0,
        atr_at_signal=1.0,
        rvol_at_signal=1.0,
        minutes_from_open=1.0,
    )
    return SignalOutcome(
        signal=sig,
        stopped_out=False,
        stop_price=99.0,
        bars_held=2,
        mfe=1.0,
        mae=0.5,
        mfe_r=1.0,
        mae_r=0.5,
        targets_hit={"origin_retest": retest},
        bars_to_target={"origin_retest": 1},
        final_return=0.01,
        exit_reason="target",
        exit_price=101.0,
        gross_r=net_r + 0.1,
        net_r=net_r,
        net_pnl_per_share=net_r,
        cost_r=0.1,
    )


def _build_outcomes(
    net_by_session: dict[int, float],
    *,
    events_per_session: int = 3,
    side: ReversalSide = ReversalSide.BULLISH,
    scenario: str = SCENARIO,
) -> list[SignalOutcome]:
    """One ordered timeline: session index -> per-event net_r (constant within a
    session), `events_per_session` events each so per-fold min_oos_events is met."""
    outs: list[SignalOutcome] = []
    for day_index in sorted(net_by_session):
        sess = _session_for(day_index)
        for _ in range(events_per_session):
            outs.append(
                _make_outcome(
                    sess,
                    net_by_session[day_index],
                    side=side,
                    scenario=scenario,
                )
            )
    return outs


# ----------------------------------------------------------------------------
# Re-derivation of the engine's own fold partition, so assertions are EXACT and
# not tautological against the implementation. This is the de Prado partition
# spelled out independently from the source comments / docstring.
# ----------------------------------------------------------------------------
def _expected_folds(n_sessions: int, cfg: WalkForwardConfig):
    """Yield (fold, oos_start, oos_end, is_end) session-index slices the way the
    documented expanding-window-with-purge scheme should carve them."""
    base = max(
        cfg.min_is_sessions,
        n_sessions - cfg.n_folds * max(1, (n_sessions - cfg.min_is_sessions) // cfg.n_folds),
    )
    remaining = n_sessions - base
    fold_size = max(1, remaining // cfg.n_folds)
    out = []
    for f in range(cfg.n_folds):
        oos_start = base + f * fold_size
        oos_end = oos_start + fold_size if f < cfg.n_folds - 1 else n_sessions
        is_end = max(0, oos_start - cfg.purge_sessions)
        out.append((f, oos_start, oos_end, is_end))
    return base, fold_size, out


def _result_for(results: dict, side: ReversalSide = ReversalSide.BULLISH) -> WalkForwardResult:
    key = (SCENARIO, side.value)
    assert key in results, f"expected key {key} in {list(results)}"
    return results[key]


# ============================================================================
# Sanity / API contract
# ============================================================================
def test_empty_input_returns_empty_dict():
    assert walk_forward([]) == {}


def test_too_few_sessions_yields_no_result():
    # min_is_sessions + n_folds is the floor; below it, _walk_forward_one bails.
    cfg = WalkForwardConfig(n_folds=5, min_is_sessions=20)
    # 24 sessions < 20 + 5 = 25 -> nothing produced.
    outs = _build_outcomes({i: 1.0 for i in range(24)})
    assert walk_forward(outs, cfg) == {}


def test_groups_split_by_scenario_and_side():
    cfg = WalkForwardConfig(n_folds=2, min_is_sessions=20, min_oos_events=1)
    a = _build_outcomes({i: 1.0 for i in range(26)}, side=ReversalSide.BULLISH, scenario="aaa")
    b = _build_outcomes({i: -1.0 for i in range(26)}, side=ReversalSide.BEARISH, scenario="bbb")
    results = walk_forward(a + b, cfg)
    assert ("aaa", "bullish") in results
    assert ("bbb", "bearish") in results
    assert len(results) == 2
    # The two groups are independent: bullish positive, bearish negative.
    assert results[("aaa", "bullish")].pooled_oos_net_r > 0
    assert results[("bbb", "bearish")].pooled_oos_net_r < 0


# ============================================================================
# NO-LOOKAHEAD: PURGE. The decisive adversarial test.
# ============================================================================
def test_purge_gap_separates_is_tail_from_oos_start():
    """For every fold, the IS window must end at least `purge_sessions` BEFORE
    the OOS fold begins. We prove it by reconstructing the engine's per-fold IS
    session count and asserting it equals (oos_start - purge_sessions), i.e. the
    session immediately before the OOS fold is DROPPED, not used in-sample."""
    cfg = WalkForwardConfig(
        n_folds=5,
        purge_sessions=1,
        embargo_sessions=1,
        min_is_sessions=20,
        min_oos_events=1,
    )
    n_sessions = 30
    outs = _build_outcomes({i: 1.0 for i in range(n_sessions)}, events_per_session=3)
    results = walk_forward(outs, cfg)
    res = _result_for(results)

    _, _, expected = _expected_folds(n_sessions, cfg)
    by_fold = {fr.fold: fr for fr in res.folds}

    for f, oos_start, oos_end, is_end in expected:
        if f not in by_fold:
            continue
        fr = by_fold[f]
        # IS uses sessions [0, is_end) where is_end = oos_start - purge_sessions.
        assert fr.is_sessions == is_end, (
            f"fold {f}: IS window had {fr.is_sessions} sessions, "
            f"expected {is_end} (= oos_start {oos_start} - purge {cfg.purge_sessions})"
        )
        # The purge actually removes a session: is_end is strictly less than
        # oos_start, so the session immediately before OOS is not in-sample.
        assert fr.is_sessions < oos_start, f"fold {f}: IS reaches the OOS start with no purge gap"
        assert fr.oos_sessions == (oos_end - oos_start)


def test_larger_purge_drops_more_is_sessions():
    """A bigger purge must shrink the IS window by exactly the extra sessions:
    purge=3 leaves 2 fewer IS sessions per fold than purge=1. Monotone, exact."""
    n_sessions = 40
    net = {i: 1.0 for i in range(n_sessions)}
    outs = _build_outcomes(net, events_per_session=3)

    cfg1 = WalkForwardConfig(n_folds=4, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    cfg3 = WalkForwardConfig(n_folds=4, purge_sessions=3, min_is_sessions=20, min_oos_events=1)
    r1 = _result_for(walk_forward(outs, cfg1))
    r3 = _result_for(walk_forward(outs, cfg3))

    by_fold1 = {fr.fold: fr for fr in r1.folds}
    by_fold3 = {fr.fold: fr for fr in r3.folds}
    common = set(by_fold1) & set(by_fold3)
    assert common, "expected overlapping folds to compare"
    for f in common:
        # purge grew by 2 sessions -> IS shrinks by exactly 2 (both have the same
        # oos_start because base/fold_size are independent of purge).
        assert by_fold1[f].is_sessions - by_fold3[f].is_sessions == 2, (
            f"fold {f}: purge delta did not shrink IS by 2"
        )


def test_purge_actually_excludes_a_poison_session_from_pooled_is():
    """Plant a huge-outlier net_r in the EXACT session that purge=1 must drop
    before fold 0's OOS. If the purge is honoured the poison never enters fold 0's
    IS window, so fold 0's IS mean stays exactly the clean 1.0. If the purge
    leaked, that one poisoned session (net_r -1000) would crush the IS mean. We
    assert the clean value, so a regression that drops the purge changes the
    asserted number -> the test fails (not a tautology).

    Geometry (n=26, n_folds=5, purge=1, min_is=20): base=21, fold_size=1, so
    fold 0's OOS is session 21 and its IS is sessions[0:20]; session 20 is the
    purge-dropped boundary session. We poison session 20. It legitimately re-enters
    LATER folds' IS (it is past data for them), so we isolate the purge by checking
    fold 0's own is_net_expectancy_r, not the pooled IS."""
    cfg = WalkForwardConfig(n_folds=5, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    n_sessions = 26
    _, _, expected = _expected_folds(n_sessions, cfg)
    fold0_oos_start = expected[0][1]  # first OOS session index (== 21)
    poison_session = fold0_oos_start - 1  # the session purge=1 must drop (20)

    net = {i: 1.0 for i in range(n_sessions)}
    net[poison_session] = -1000.0  # outlier ONLY in the purged slot
    outs = _build_outcomes(net, events_per_session=3)
    res = _result_for(walk_forward(outs, cfg))

    by_fold = {fr.fold: fr for fr in res.folds}
    assert 0 in by_fold, "fold 0 should survive (IS=20 >= min_is=20)"
    fr0 = by_fold[0]
    # Fold 0 IS = sessions[0, poison_session): all net_r == 1.0, none poisoned.
    assert fr0.is_sessions == poison_session  # 20 IS sessions, session 20 dropped
    assert fr0.is_net_expectancy_r == pytest.approx(1.0, abs=1e-9), (
        "fold 0 IS mean is not the clean 1.0 -> the purged poison session (the "
        "boundary session immediately before fold 0's OOS) leaked into the "
        f"in-sample window (got {fr0.is_net_expectancy_r})"
    )


def test_no_fold_uses_its_own_oos_sessions_in_sample():
    """The strongest no-lookahead invariant: the IS window of fold f must not
    overlap fold f's OOS span. We verify is_sessions (the count of IS sessions,
    all drawn from the timeline head) never reaches into the OOS block."""
    cfg = WalkForwardConfig(n_folds=5, purge_sessions=2, min_is_sessions=20, min_oos_events=1)
    n_sessions = 35
    outs = _build_outcomes({i: 1.0 for i in range(n_sessions)}, events_per_session=2)
    res = _result_for(walk_forward(outs, cfg))
    _, _, expected = _expected_folds(n_sessions, cfg)
    by_fold = {fr.fold: fr for fr in res.folds}
    for f, oos_start, _oos_end, _is_end in expected:
        if f not in by_fold:
            continue
        # IS sessions are sessions[0:is_end]; the OOS block is [oos_start:oos_end].
        # is_end <= oos_start - purge < oos_start, so the IS block ends strictly
        # before the OOS block begins. Assert it directly.
        assert by_fold[f].is_sessions <= oos_start - cfg.purge_sessions


# ============================================================================
# NO-LOOKAHEAD: EMBARGO. Documented + configured but NOT implemented.
# ============================================================================
@pytest.mark.xfail(
    reason=(
        "ENGINE BUG: embargo_sessions is documented (module docstring + "
        "WalkForwardConfig comment 'sessions skipped after each OOS fold') and "
        "configurable, but `_walk_forward_one` never references it. The IS window "
        "for fold f is sessions[0 : oos_start - purge_sessions] regardless of "
        "embargo, so an OOS fold's sessions re-enter the very next fold's IS "
        "window with NO embargo gap. de Prado embargo leakage guard is absent. "
        "Evidence: grep 'embargo' walkforward.py -> only docstring/comment/config."
    ),
    strict=False,
)
def test_embargo_gap_is_honored_between_oos_and_next_is():
    """After fold f's OOS block ends, fold f+1's IS window should resume only
    after an embargo gap of `embargo_sessions`. Construct a case where a large
    embargo MUST shrink the later folds' IS windows relative to a zero embargo.
    If embargo were implemented, embargo=3 would yield strictly smaller IS counts
    on folds >= 1 than embargo=0. It does not, so this xfails."""
    n_sessions = 40
    net = {i: 1.0 for i in range(n_sessions)}
    outs = _build_outcomes(net, events_per_session=3)

    cfg0 = WalkForwardConfig(
        n_folds=4,
        purge_sessions=1,
        embargo_sessions=0,
        min_is_sessions=20,
        min_oos_events=1,
    )
    cfg3 = WalkForwardConfig(
        n_folds=4,
        purge_sessions=1,
        embargo_sessions=3,
        min_is_sessions=20,
        min_oos_events=1,
    )
    r0 = _result_for(walk_forward(outs, cfg0))
    r3 = _result_for(walk_forward(outs, cfg3))
    by0 = {fr.fold: fr for fr in r0.folds}
    by3 = {fr.fold: fr for fr in r3.folds}

    later = [f for f in (set(by0) & set(by3)) if f >= 1]
    assert later, "need folds >= 1 to observe an embargo effect"
    # A real embargo on fold f>=1 would EXCLUDE the embargo_sessions immediately
    # after the prior OOS fold from this fold's IS window -> strictly fewer IS
    # sessions. We assert that here; the engine ignores embargo so IS is identical
    # and this assertion fails (xfail).
    assert any(by3[f].is_sessions < by0[f].is_sessions for f in later), (
        "embargo_sessions had NO effect on any later fold's IS window count -> "
        "the embargo leakage guard is not implemented"
    )


# ============================================================================
# Aggregation correctness on KNOWN inputs.
# ============================================================================
def test_pooled_oos_mean_matches_planted_constant():
    """All net_r == 0.7 everywhere -> pooled OOS net expectancy must be exactly
    0.7 (bootstrap mean of a constant is the constant), decay ~ 0."""
    cfg = WalkForwardConfig(n_folds=4, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    outs = _build_outcomes({i: 0.7 for i in range(32)}, events_per_session=3)
    res = _result_for(walk_forward(outs, cfg))
    assert res.pooled_oos_net_r == pytest.approx(0.7, abs=1e-9)
    assert res.pooled_is_net_r == pytest.approx(0.7, abs=1e-9)
    assert res.decay_r == pytest.approx(0.0, abs=1e-9)
    lo, hi = res.pooled_oos_ci
    assert lo == pytest.approx(0.7, abs=1e-9) and hi == pytest.approx(0.7, abs=1e-9)


def test_decay_detects_overfit_is_to_oos_drop():
    """IS sessions strongly positive, OOS sessions strongly negative -> a large
    positive decay (IS - OOS). This is the engine's overfit gauge; a high decay
    must surface when the edge does not generalise."""
    cfg = WalkForwardConfig(n_folds=4, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    n_sessions = 32
    _, _, expected = _expected_folds(n_sessions, cfg)
    first_oos = expected[0][1]
    # Early (IS-heavy) sessions positive, the OOS tail negative.
    net = {}
    for i in range(n_sessions):
        net[i] = 2.0 if i < first_oos else -2.0
    outs = _build_outcomes(net, events_per_session=3)
    res = _result_for(walk_forward(outs, cfg))
    assert res.pooled_is_net_r > 0
    assert res.pooled_oos_net_r < 0
    assert res.decay_r > 0  # IS minus OOS: positive == decayed edge


def test_trustworthy_requires_oos_ci_clear_of_zero():
    """trustworthy is True only when pooled OOS CI lower bound > 0 AND enough
    events. A uniformly strong-positive edge with ample events is trustworthy; a
    zero-mean edge is not."""
    cfg = WalkForwardConfig(n_folds=4, purge_sessions=1, min_is_sessions=20, min_oos_events=5)
    good = _build_outcomes({i: 0.9 for i in range(32)}, events_per_session=4)
    res_good = _result_for(walk_forward(good, cfg))
    assert res_good.trustworthy is True
    assert res_good.pooled_oos_ci[0] > 0

    # All net_r exactly 0 -> CI is [0, 0], lower bound not > 0 -> untrustworthy.
    flat = _build_outcomes({i: 0.0 for i in range(32)}, events_per_session=4)
    res_flat = _result_for(walk_forward(flat, cfg))
    assert res_flat.trustworthy is False


def test_n_oos_events_equals_sum_of_fold_oos_events():
    """The pooled OOS event count must equal the sum of per-fold OOS events --
    no event double-counted across folds, none dropped. With contiguous,
    non-overlapping OOS blocks each event appears in exactly one fold."""
    cfg = WalkForwardConfig(n_folds=5, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    outs = _build_outcomes({i: 1.0 for i in range(30)}, events_per_session=3)
    res = _result_for(walk_forward(outs, cfg))
    assert res.n_oos_events == sum(fr.oos_events for fr in res.folds)
    # Every OOS session contributes exactly events_per_session events.
    for fr in res.folds:
        assert fr.oos_events == 3 * fr.oos_sessions


def test_min_oos_events_skips_thin_folds():
    """A fold with fewer than min_oos_events OOS events must be skipped, not
    reported. Plant 1 event/session and demand 5/fold; folds of 1-session each
    (fold_size==1) hold 1 event and must be dropped, yielding no usable folds."""
    cfg = WalkForwardConfig(n_folds=5, purge_sessions=1, min_is_sessions=20, min_oos_events=5)
    # 25 sessions -> fold_size 1, 1 event/session => every fold has 1 OOS event
    # < 5 => all folds skipped => no result for the group at all.
    outs = _build_outcomes({i: 1.0 for i in range(25)}, events_per_session=1)
    results = walk_forward(outs, cfg)
    assert results == {}, "folds below min_oos_events should be skipped, leaving no result"


def test_determinism_same_input_same_output():
    """Bootstrap CIs are seeded -> identical inputs give identical pooled results
    across runs. Non-determinism here would make every downstream report unstable."""
    cfg = WalkForwardConfig(n_folds=4, purge_sessions=1, min_is_sessions=20, min_oos_events=1)
    net = {i: float((i % 7) - 3) * 0.3 for i in range(34)}  # mixed signs, fixed
    outs = _build_outcomes(net, events_per_session=3)
    r1 = _result_for(walk_forward(outs, cfg))
    r2 = _result_for(walk_forward(outs, cfg))
    assert r1.pooled_oos_net_r == r2.pooled_oos_net_r
    assert r1.pooled_oos_ci == r2.pooled_oos_ci
    assert r1.n_oos_events == r2.n_oos_events
    assert [f.oos_events for f in r1.folds] == [f.oos_events for f in r2.folds]
