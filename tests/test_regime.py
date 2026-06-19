"""
Adversarial tests for engine/intraday/regime.py and regime_conditioning.py.

The Prime Directive (from _synth.py): tests must actively try to BREAK the
engine. Here we plant UNAMBIGUOUS regimes and assert the classifier returns the
planted label exactly -- no "it ran" tautologies.

Strategy:
    * Clean monotone uptrend  -> TREND_UP  (efficiency ratio == 1.0, slope > 0).
    * Clean monotone downtrend -> TREND_DOWN.
    * Flat sawtooth oscillation -> RANGE (efficiency ~0, far below threshold).
    * Wide-but-choppy day      -> RANGE despite a wide range (low efficiency must
                                  veto a false trend) -- the documented invariant.
    * Volatility: session range vs an EXPLICIT baseline -> HIGH / NORMAL / LOW vol
                  at hand-computed ratios straddling the 1.5 / 0.6 thresholds.
    * No-lookahead: classify_all_sessions baselines on PRIOR sessions only;
                    corrupting FUTURE sessions must not change an earlier label.

The regime functions operate on plain OHLC DataFrames keyed by a `date` column
(the per-row session identifier), so we build DataFrames directly with the exact
columns the module reads (date, open, high, low, close) for full control.

Public API exercised:
    efficiency_ratio(close: np.ndarray) -> float
    classify_session_regime(session_bars, session_atr, vol_baseline_atr_range,
        trend_eff_threshold=0.45, trend_slope_atr_threshold=0.02,
        high_vol_mult=1.5, low_vol_mult=0.6) -> SessionRegime
    classify_all_sessions(bars, atr_lookback_sessions=14)
        -> dict[Timestamp, SessionRegime]
    DirectionalRegime, VolatilityRegime, SessionRegime(.label, ...)
    condition_on_regime(outcomes, regimes, by_volatility=True)
        -> dict[(scenario, side, regime_label), RegimeEdge]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.intraday.backtest import SignalOutcome
from engine.intraday.regime import (
    DirectionalRegime,
    SessionRegime,
    VolatilityRegime,
    classify_all_sessions,
    classify_session_regime,
    efficiency_ratio,
)
from engine.intraday.regime_conditioning import condition_on_regime
from engine.intraday.reversal import ReversalSide, ReversalSignal

DATE = "2026-06-01"


def _df_from_closes(closes, *, date: str = DATE, wick: float = 0.01) -> pd.DataFrame:
    """OHLC frame from a close path: open[i]=close[i-1] (continuous tape), the
    high/low extend a tiny `wick` beyond the body so the session range tracks the
    close path. A `date` column tags every row with the session it belongs to."""
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * n,
            "open": opens,
            "high": body_hi + wick,
            "low": body_lo - wick,
            "close": closes,
        }
    )


# ============================================================================
# efficiency_ratio: the trend-vs-range workhorse.
# ============================================================================
def test_efficiency_ratio_perfect_trend_is_one():
    # Monotone path: |net| == sum(|steps|) -> ratio exactly 1.0.
    closes = np.arange(0.0, 10.0, 1.0)
    assert efficiency_ratio(closes) == pytest.approx(1.0, abs=1e-12)


def test_efficiency_ratio_round_trip_is_zero():
    # Up then back to start: net 0 -> ratio 0.0 regardless of path length.
    closes = np.array([0.0, 5.0, 0.0])
    assert efficiency_ratio(closes) == pytest.approx(0.0, abs=1e-12)


def test_efficiency_ratio_sawtooth_is_small():
    # 15 up-down cycles: net is one step, path is 30 steps -> ~1/29.
    closes = np.array([100.0, 101.0] * 15)
    er = efficiency_ratio(closes)
    assert 0.0 < er < 0.1


def test_efficiency_ratio_degenerate_inputs():
    assert efficiency_ratio(np.array([5.0])) == 0.0  # < 2 points
    assert efficiency_ratio(np.array([7.0, 7.0, 7.0])) == 0.0  # zero path


# ============================================================================
# Directional regime: planted trends and ranges.
# ============================================================================
def test_clean_uptrend_classified_trend_up():
    closes = np.arange(100.0, 130.0, 1.0)  # +1/bar, 30 bars
    reg = classify_session_regime(
        _df_from_closes(closes), session_atr=1.0, vol_baseline_atr_range=10.0
    )
    assert reg.directional is DirectionalRegime.TREND_UP
    assert reg.efficiency_ratio == pytest.approx(1.0, abs=1e-9)
    assert reg.regression_slope_atr > 0
    assert reg.r_squared == pytest.approx(1.0, abs=1e-9)


def test_clean_downtrend_classified_trend_down():
    closes = np.arange(130.0, 100.0, -1.0)  # -1/bar
    reg = classify_session_regime(
        _df_from_closes(closes), session_atr=1.0, vol_baseline_atr_range=10.0
    )
    assert reg.directional is DirectionalRegime.TREND_DOWN
    assert reg.regression_slope_atr < 0


def test_flat_oscillation_classified_range():
    closes = np.array([100.0, 101.0] * 15)  # sawtooth, net ~0
    reg = classify_session_regime(
        _df_from_closes(closes), session_atr=1.0, vol_baseline_atr_range=2.0
    )
    assert reg.directional is DirectionalRegime.RANGE
    assert reg.efficiency_ratio < 0.45


def test_wide_but_choppy_is_range_not_false_trend():
    """The documented invariant: a wide-but-choppy day must be RANGE, not a false
    trend. Big +-10 swings give a WIDE range but the net move is ~0, so the
    efficiency ratio vetoes the trend call even though the slope magnitude alone
    might pass. This is the adversarial case the classifier is designed to reject."""
    closes = np.array([100.0, 110.0] * 6)  # huge swings, net 0
    reg = classify_session_regime(
        _df_from_closes(closes), session_atr=1.0, vol_baseline_atr_range=5.0
    )
    assert reg.directional is DirectionalRegime.RANGE, (
        f"wide-choppy day misclassified as {reg.directional} (a false trend)"
    )
    assert reg.range_atr > 5.0  # genuinely a wide day...
    assert reg.efficiency_ratio < 0.45  # ...but inefficient -> RANGE


def test_trend_needs_both_efficiency_and_slope():
    """A perfectly efficient but microscopically-sloped path must NOT be a trend:
    efficiency==1 but slope_atr below trend_slope_atr_threshold -> RANGE. Proves
    the AND gate (both conditions), not just one."""
    # 30 bars rising by 0.001 each: efficiency 1.0, but slope/atr = 0.001 < 0.02.
    closes = 100.0 + np.arange(30) * 0.001
    reg = classify_session_regime(
        _df_from_closes(closes, wick=0.0),
        session_atr=1.0,
        vol_baseline_atr_range=10.0,
    )
    assert reg.efficiency_ratio == pytest.approx(1.0, abs=1e-9)
    assert abs(reg.regression_slope_atr) < 0.02
    assert reg.directional is DirectionalRegime.RANGE


def test_custom_thresholds_flip_the_call():
    """A path that is RANGE at the default slope threshold becomes TREND_UP if the
    caller lowers trend_slope_atr_threshold below the path's slope. Proves the
    threshold params are actually wired into the decision."""
    closes = 100.0 + np.arange(30) * 0.005  # slope_atr 0.005, eff 1.0
    bars = _df_from_closes(closes, wick=0.0)
    default = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=10.0)
    assert default.directional is DirectionalRegime.RANGE  # 0.005 < 0.02

    loosened = classify_session_regime(
        bars,
        session_atr=1.0,
        vol_baseline_atr_range=10.0,
        trend_slope_atr_threshold=0.001,
    )
    assert loosened.directional is DirectionalRegime.TREND_UP


# ============================================================================
# Volatility regime: explicit baseline, hand-computed ratios.
# ============================================================================
def _wide_flat_session(swing: float = 10.0, wick: float = 0.0) -> pd.DataFrame:
    """A 16-bar sawtooth between 100 and 100+swing: range_atr ~= swing (with
    session_atr=1.0), efficiency ~0 (so directional == RANGE), letting us isolate
    the volatility axis from the directional axis."""
    return _df_from_closes(np.array([100.0, 100.0 + swing] * 8), wick=wick)


def test_high_vol_when_range_exceeds_baseline():
    bars = _wide_flat_session(swing=10.0)
    # range_atr ~= 10; baseline 5 -> ratio ~2.0 >= 1.5 -> HIGH_VOL.
    reg = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=5.0)
    assert reg.volatility is VolatilityRegime.HIGH_VOL


def test_normal_vol_at_baseline():
    bars = _wide_flat_session(swing=10.0)
    # range_atr ~= 10; baseline 10 -> ratio ~1.0 -> NORMAL_VOL.
    reg = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=10.0)
    assert reg.volatility is VolatilityRegime.NORMAL_VOL


def test_low_vol_when_range_well_below_baseline():
    bars = _wide_flat_session(swing=10.0)
    # range_atr ~= 10; baseline 20 -> ratio ~0.5 <= 0.6 -> LOW_VOL.
    reg = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=20.0)
    assert reg.volatility is VolatilityRegime.LOW_VOL


def test_volatility_axis_independent_of_direction():
    """A clean UPTREND can be LOW_VOL: directional and volatility are orthogonal
    axes. Same uptrend, two baselines -> TREND_UP in both, vol flips."""
    closes = np.arange(100.0, 110.0, 0.5)  # 20 bars, net 9.5, range ~9.5
    bars = _df_from_closes(closes)
    hi = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=4.0)
    lo = classify_session_regime(bars, session_atr=1.0, vol_baseline_atr_range=30.0)
    assert hi.directional is DirectionalRegime.TREND_UP
    assert lo.directional is DirectionalRegime.TREND_UP
    assert hi.volatility is VolatilityRegime.HIGH_VOL
    assert lo.volatility is VolatilityRegime.LOW_VOL


def test_label_property_combines_both_axes():
    closes = np.arange(100.0, 130.0, 1.0)
    reg = classify_session_regime(
        _df_from_closes(closes), session_atr=1.0, vol_baseline_atr_range=5.0
    )
    assert reg.label == f"{reg.directional.value}/{reg.volatility.value}"
    assert "/" in reg.label


# ============================================================================
# classify_all_sessions: multi-session and NO-LOOKAHEAD.
# ============================================================================
def _multi_session_frame() -> pd.DataFrame:
    frames = [
        _df_from_closes(np.arange(100.0, 130.0, 1.0), date="2026-06-01"),  # up
        _df_from_closes(np.array([100.0, 101.0] * 15), date="2026-06-02"),  # range
        _df_from_closes(np.arange(130.0, 100.0, -1.0), date="2026-06-03"),  # down
    ]
    return pd.concat(frames, ignore_index=True)


def test_classify_all_sessions_labels_each_planted_session():
    res = classify_all_sessions(_multi_session_frame())
    assert res[pd.Timestamp("2026-06-01")].directional is DirectionalRegime.TREND_UP
    assert res[pd.Timestamp("2026-06-02")].directional is DirectionalRegime.RANGE
    assert res[pd.Timestamp("2026-06-03")].directional is DirectionalRegime.TREND_DOWN


def test_classify_all_sessions_skips_short_sessions():
    """A session with < 5 bars is dropped (cannot estimate a regime from it)."""
    long_sess = _df_from_closes(np.arange(100.0, 130.0, 1.0), date="2026-06-01")
    short_sess = _df_from_closes(np.array([100.0, 101.0, 102.0, 103.0]), date="2026-06-02")
    bars = pd.concat([long_sess, short_sess], ignore_index=True)
    res = classify_all_sessions(bars)
    assert pd.Timestamp("2026-06-01") in res
    assert pd.Timestamp("2026-06-02") not in res


def test_classify_all_sessions_no_lookahead_in_baseline():
    """The CAUSALITY PROBE: the volatility baseline for session t uses PRIOR
    sessions only. Corrupt every FUTURE session into an enormous-range day; an
    EARLIER session's classification must be byte-for-byte identical. Any change
    is a lookahead leak (the future baseline bled backwards)."""

    def build(future_swing: float) -> pd.DataFrame:
        frames = []
        # Six prior baseline sessions + a target session, all identical width.
        for i in range(7):
            frames.append(
                _df_from_closes(np.array([100.0, 110.0] * 8), date=f"2026-06-{i + 1:02d}")
            )
        # We assert on session "2026-06-07" (index 6).
        # Three FUTURE sessions whose width is scaled by future_swing.
        for j in range(3):
            frames.append(
                _df_from_closes(
                    np.array([100.0, 100.0 + future_swing] * 8),
                    date=f"2026-06-{8 + j:02d}",
                )
            )
        return pd.concat(frames, ignore_index=True), pd.Timestamp("2026-06-07")

    clean_bars, target = build(future_swing=10.0)
    corrupt_bars, _ = build(future_swing=500.0)  # absurdly wide FUTURE sessions

    clean = classify_all_sessions(clean_bars)[target]
    corrupt = classify_all_sessions(corrupt_bars)[target]

    assert clean == corrupt, (
        "an earlier session's regime changed when FUTURE sessions were corrupted "
        "-> the volatility baseline leaked future information (lookahead)"
    )


def test_first_session_bootstraps_its_own_baseline():
    """With no prior history the first session uses its own range as baseline, so
    ratio == 1.0 -> NORMAL_VOL. Deterministic, not noise-dependent."""
    res = classify_all_sessions(_multi_session_frame())
    first = res[pd.Timestamp("2026-06-01")]
    assert first.volatility is VolatilityRegime.NORMAL_VOL


# ============================================================================
# regime_conditioning.condition_on_regime
# ============================================================================
def _mk_signal(session: str, side: ReversalSide, scenario: str) -> ReversalSignal:
    ts = pd.Timestamp(session)
    return ReversalSignal(
        symbol="TEST",
        session=ts,
        side=side,
        scenario=scenario,
        signal_time=ts + pd.Timedelta(hours=9, minutes=31),
        signal_index=1,
        entry_price=100.0,
        origin_extreme=101.0,
        counter_extreme=99.0,
        vwap_at_signal=100.0,
        atr_at_signal=1.0,
        rvol_at_signal=1.0,
        minutes_from_open=1.0,
    )


def _mk_outcome(
    session: str,
    net_r: float,
    retest: bool,
    *,
    side: ReversalSide = ReversalSide.BULLISH,
    scenario: str = "sc",
) -> SignalOutcome:
    return SignalOutcome(
        signal=_mk_signal(session, side, scenario),
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


def _mk_regime(session: str, d: DirectionalRegime, v: VolatilityRegime) -> SessionRegime:
    return SessionRegime(
        session=pd.Timestamp(session),
        directional=d,
        volatility=v,
        efficiency_ratio=0.8,
        regression_slope_atr=0.5,
        range_atr=2.0,
        r_squared=0.9,
    )


def test_condition_on_regime_buckets_by_combined_label():
    regimes = {
        pd.Timestamp("2026-06-01"): _mk_regime(
            "2026-06-01", DirectionalRegime.TREND_UP, VolatilityRegime.HIGH_VOL
        ),
        pd.Timestamp("2026-06-02"): _mk_regime(
            "2026-06-02", DirectionalRegime.RANGE, VolatilityRegime.LOW_VOL
        ),
    }
    outs = [
        _mk_outcome("2026-06-01", 1.0, True),
        _mk_outcome("2026-06-01", 2.0, True),
        _mk_outcome("2026-06-02", -1.0, False),
    ]
    res = condition_on_regime(outs, regimes, by_volatility=True)
    trend_key = ("sc", "bullish", "trend_up/high_vol")
    range_key = ("sc", "bullish", "range/low_vol")
    assert set(res) == {trend_key, range_key}
    assert res[trend_key].n == 2
    assert res[range_key].n == 1
    # net expectancy per bucket reflects the planted net_r (mean of constants).
    assert res[trend_key].net_expectancy_r == pytest.approx(1.5, abs=1e-9)
    assert res[range_key].net_expectancy_r == pytest.approx(-1.0, abs=1e-9)


def test_condition_on_regime_directional_only_collapses_vol():
    """by_volatility=False must label by directional regime ALONE, merging
    sessions that differ only in volatility into one coarser bucket."""
    regimes = {
        pd.Timestamp("2026-06-01"): _mk_regime(
            "2026-06-01", DirectionalRegime.TREND_UP, VolatilityRegime.HIGH_VOL
        ),
        pd.Timestamp("2026-06-02"): _mk_regime(
            "2026-06-02", DirectionalRegime.TREND_UP, VolatilityRegime.LOW_VOL
        ),
    }
    outs = [
        _mk_outcome("2026-06-01", 1.0, True),
        _mk_outcome("2026-06-02", 3.0, True),
    ]
    res = condition_on_regime(outs, regimes, by_volatility=False)
    key = ("sc", "bullish", "trend_up")
    assert set(res) == {key}  # both vol regimes merged
    assert res[key].n == 2
    assert res[key].net_expectancy_r == pytest.approx(2.0, abs=1e-9)


def test_condition_on_regime_drops_outcomes_without_regime():
    """An outcome whose session has no entry in `regimes` is dropped, not crashed
    on, not bucketed under a phantom label."""
    regimes = {
        pd.Timestamp("2026-06-01"): _mk_regime(
            "2026-06-01", DirectionalRegime.RANGE, VolatilityRegime.NORMAL_VOL
        ),
    }
    outs = [
        _mk_outcome("2026-06-01", 1.0, True),
        _mk_outcome("2026-06-09", 5.0, True),  # no regime for this session
    ]
    res = condition_on_regime(outs, regimes)
    assert len(res) == 1
    only = next(iter(res.values()))
    assert only.n == 1  # the orphan outcome is excluded


def test_condition_on_regime_separates_scenario_and_side():
    """Distinct (scenario, side) pairs in the SAME regime become distinct buckets:
    the conditioning must not blend a bullish edge with a bearish one."""
    regimes = {
        pd.Timestamp("2026-06-01"): _mk_regime(
            "2026-06-01", DirectionalRegime.TREND_UP, VolatilityRegime.NORMAL_VOL
        ),
    }
    outs = [
        _mk_outcome("2026-06-01", 1.0, True, side=ReversalSide.BULLISH, scenario="a"),
        _mk_outcome("2026-06-01", -1.0, False, side=ReversalSide.BEARISH, scenario="a"),
        _mk_outcome("2026-06-01", 2.0, True, side=ReversalSide.BULLISH, scenario="b"),
    ]
    res = condition_on_regime(outs, regimes, by_volatility=True)
    label = "trend_up/normal_vol"
    assert ("a", "bullish", label) in res
    assert ("a", "bearish", label) in res
    assert ("b", "bullish", label) in res
    assert len(res) == 3


def test_condition_on_regime_empty_inputs():
    assert condition_on_regime([], {}) == {}
    # outcomes present but no regimes at all -> nothing joins -> empty.
    outs = [_mk_outcome("2026-06-01", 1.0, True)]
    assert condition_on_regime(outs, {}) == {}
