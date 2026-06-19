"""
Adversarial tests for engine.session.session.Session.

These tests try to BREAK the Session contract: they smuggle pre/post-market
bars past the regular-hours filter, corrupt the future to expose lookahead in
VWAP, hand-compute mean true range to catch a regression to the documented
range/sqrt(n) bug, and verify running extremes are causal and monotone.

All randomness is seeded via np.random.default_rng(SEED). No tautologies: every
assertion is checked against an independently hand-computed expectation.
"""

from __future__ import annotations

import math

import _synth as S
import numpy as np
import pandas as pd
import pytest

from engine.session.session import Session

SEED = 20260618


# ----------------------------------------------------------------------------
# 1. Regular-hours filter drops pre/post-market bars
# ----------------------------------------------------------------------------
def test_regular_hours_filter_drops_premarket_and_postmarket():
    """A 09:00 pre-market spike and a 16:30 post-market spike must NOT bleed
    into session high/low/vwap. Only [09:30, 16:00) survives."""
    ts = pd.to_datetime(
        [
            "2026-06-01 09:00",  # pre-market: high spike that must be dropped
            "2026-06-01 09:30",
            "2026-06-01 09:31",
            "2026-06-01 09:32",
            "2026-06-01 09:33",
            "2026-06-01 09:34",
            "2026-06-01 16:30",  # post-market: low spike that must be dropped
        ]
    )
    df = pd.DataFrame(
        {
            "datetime": ts,
            "open": [50.0, 100.0, 101.0, 102.0, 103.0, 104.0, 9999.0],
            "high": [50.0, 200.0, 101.0, 102.0, 103.0, 104.0, 9999.0],
            # pre-market low is -5 (would corrupt session low if not filtered)
            "low": [-5.0, 100.0, 101.0, 102.0, 103.0, 104.0, -999.0],
            "close": [50.0, 100.0, 101.0, 102.0, 103.0, 104.0, 9999.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    s = Session.from_intraday("X", df, min_bars=5)

    # Exactly the 5 regular-hours bars survive.
    assert len(s) == 5
    # High/low come ONLY from the regular session, not the pre/post spikes.
    assert s.high == 200.0  # from the 09:30 bar, not 9999
    assert s.low == 100.0  # from the regular bars, not -5 or -999
    # First/last bars are the regular open and the last regular bar.
    assert s.bars["datetime"].iloc[0].strftime("%H:%M") == "09:30"
    assert s.bars["datetime"].iloc[-1].strftime("%H:%M") == "09:34"
    # VWAP reflects only the surviving bars: typical prices are all within
    # [100, 200], so vwap is strictly inside that band (a leaked -5 or 9999
    # would push it outside).
    assert 100.0 <= s.vwap.iloc[-1] <= 200.0


def test_sixteen_hundred_is_excluded_endpoint():
    """16:00 itself is OUTSIDE the half-open [09:30, 16:00) window."""
    ts = pd.to_datetime(
        [
            "2026-06-01 15:56",
            "2026-06-01 15:57",
            "2026-06-01 15:58",
            "2026-06-01 15:59",
            "2026-06-01 16:00",  # must be dropped
        ]
    )
    df = pd.DataFrame(
        {
            "datetime": ts,
            "open": [1.0, 2.0, 3.0, 4.0, 9999.0],
            "high": [1.0, 2.0, 3.0, 4.0, 9999.0],
            "low": [1.0, 2.0, 3.0, 4.0, 9999.0],
            "close": [1.0, 2.0, 3.0, 4.0, 9999.0],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    s = Session.from_intraday("X", df, min_bars=4)
    assert len(s) == 4
    assert s.high == 4.0  # the 16:00 bar (9999) is excluded
    assert s.bars["datetime"].iloc[-1].strftime("%H:%M") == "15:59"


# ----------------------------------------------------------------------------
# 2. from_intraday validation
# ----------------------------------------------------------------------------
def test_from_intraday_raises_on_missing_required_column():
    df = S.bars_from_closes([100, 101, 102, 103, 104, 105]).drop(columns=["high"])
    with pytest.raises(ValueError) as ei:
        Session.from_intraday("X", df)
    # The error names the missing column.
    assert "high" in str(ei.value)


def test_from_intraday_raises_on_each_missing_required_column():
    """Every required column (datetime/open/high/low/close), individually,
    triggers a ValueError. volume is NOT required and must not raise."""
    base = S.bars_from_closes([100, 101, 102, 103, 104, 105])
    for col in ("datetime", "open", "high", "low", "close"):
        with pytest.raises(ValueError):
            Session.from_intraday("X", base.drop(columns=[col]))
    # Dropping volume is fine (it is synthesized as NaN).
    s = Session.from_intraday("X", base.drop(columns=["volume"]))
    assert len(s) == 6


def test_from_intraday_raises_below_min_bars():
    """Fewer than min_bars regular-hours bars must raise (not silently build)."""
    df = S.bars_from_closes([100, 101, 102])  # 3 bars
    with pytest.raises(ValueError) as ei:
        Session.from_intraday("X", df, min_bars=5)
    assert "3" in str(ei.value) and "5" in str(ei.value)


def test_from_intraday_raises_when_all_bars_filtered_out():
    """If every bar is outside regular hours, construction must raise."""
    ts = pd.to_datetime(
        [
            "2026-06-01 08:00",
            "2026-06-01 08:01",
            "2026-06-01 08:02",
            "2026-06-01 08:03",
            "2026-06-01 08:04",
        ]
    )
    df = pd.DataFrame(
        {
            "datetime": ts,
            "open": [1.0] * 5,
            "high": [1.0] * 5,
            "low": [1.0] * 5,
            "close": [1.0] * 5,
            "volume": [1.0] * 5,
        }
    )
    with pytest.raises(ValueError):
        Session.from_intraday("X", df, min_bars=1)


# ----------------------------------------------------------------------------
# 3. VWAP is causal and genuinely volume-weighted
# ----------------------------------------------------------------------------
def test_vwap_is_causal_under_future_corruption():
    """vwap.iloc[k] must depend only on bars[0:k+1]. Corrupt every bar after k
    (S.corrupt_future) and assert vwap[:k+1] is bit-for-bit unchanged. A
    lookahead leak would change a past value."""
    df = S.bars_from_closes([100, 101, 99, 103, 105, 102, 107, 110, 108, 112], wick=0.2, vol=1000.0)
    s_clean = Session.from_intraday("X", df)
    for k in (0, 1, 4, 7):
        corrupt = S.corrupt_future(df, k)
        s_corrupt = Session.from_intraday("X", corrupt)
        np.testing.assert_allclose(
            s_clean.vwap.iloc[: k + 1].to_numpy(),
            s_corrupt.vwap.iloc[: k + 1].to_numpy(),
            rtol=0,
            atol=0,
            err_msg=f"vwap[:{k + 1}] changed when future bars were corrupted",
        )


def test_vwap_is_volume_weighted_not_plain_typical_mean():
    """Build a case where volume-weighting gives a DIFFERENT answer than a plain
    typical-price mean, and assert vwap matches the volume-weighted value."""
    # Bar0 typical=100 vol=1000; Bar1 typical=110 vol=9000; remaining bars
    # typical=100 vol=0 (neutral, do not move the cumulative VWAP).
    ts = pd.date_range("2026-06-01 09:30", periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "datetime": ts,
            "open": [100.0, 110.0, 100.0, 100.0, 100.0],
            "high": [100.0, 110.0, 100.0, 100.0, 100.0],
            "low": [100.0, 110.0, 100.0, 100.0, 100.0],
            "close": [100.0, 110.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 9000.0, 0.0, 0.0, 0.0],
        }
    )
    s = Session.from_intraday("X", df, min_bars=5)
    assert s.vwap_is_volume_weighted is True

    # Hand-computed volume-weighted VWAP at bar 1:
    expected_vw = (100.0 * 1000.0 + 110.0 * 9000.0) / (1000.0 + 9000.0)  # 109.0
    plain_mean = (100.0 + 110.0) / 2.0  # 105.0 -- the WRONG (unweighted) answer
    assert expected_vw != plain_mean  # the two genuinely differ
    assert s.vwap.iloc[1] == pytest.approx(expected_vw)
    assert s.vwap.iloc[1] != pytest.approx(plain_mean)
    # Zero-volume bars after must not move the cumulative VWAP.
    assert s.vwap.iloc[-1] == pytest.approx(expected_vw)


def test_vwap_falls_back_to_typical_mean_without_volume():
    """When volume is entirely absent, vwap is the expanding mean of typical."""
    df = S.bars_from_closes([100, 102, 104, 106, 108], wick=0.0).drop(columns=["volume"])
    s = Session.from_intraday("X", df)
    assert s.vwap_is_volume_weighted is False
    b = s.bars
    typical = (b["high"] + b["low"] + b["close"]) / 3.0
    expected = typical.expanding().mean().reset_index(drop=True)
    np.testing.assert_allclose(s.vwap.to_numpy(), expected.to_numpy())


# ----------------------------------------------------------------------------
# 4. atr_mean == mean true range, and NOT range/sqrt(n)
# ----------------------------------------------------------------------------
def test_atr_mean_equals_tr_on_constant_tr_session():
    """On constant_tr_session every bar TR == tr, so atr_mean == tr exactly."""
    for tr in (0.5, 1.0, 2.0):
        s = S.constant_tr_session(30, base=100.0, tr=tr)
        assert s.atr_mean == pytest.approx(tr, abs=1e-12)


def test_atr_mean_equals_handcomputed_mean_tr():
    """Hand-built 3-bar frame: atr_mean must equal the hand-computed mean TR
    using prev_close = close.shift(1).fillna(close)."""
    opens = [100.0, 101.0, 102.0]
    highs = [101.5, 102.5, 102.5]
    lows = [99.5, 100.5, 100.5]
    closes = [101.0, 102.0, 101.0]
    df = S.make_intraday_df(opens, highs, lows, closes)
    s = Session.from_intraday("X", df, min_bars=3)

    # Hand compute mean TR. Bar0 prev_close = close[0] = 101.0 (fillna).
    prev = [101.0, 101.0, 102.0]
    trs = []
    for h, lo, pc in zip(highs, lows, prev):
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    expected = sum(trs) / len(trs)
    assert expected == pytest.approx(2.0)  # sanity: each bar TR is 2.0
    assert s.atr_mean == pytest.approx(expected, abs=1e-12)


def test_atr_mean_does_not_equal_range_over_sqrt_n():
    """Guards the documented prior bug (flooring with range/sqrt(bars), which
    forces range/atr ~= sqrt(n)). Construct cases where mean-TR and
    range/sqrt(n) differ, and assert atr_mean tracks mean-TR."""
    # Case A: flat constant-TR session. range == tr, so range/sqrt(n) << tr.
    s = S.constant_tr_session(36, base=100.0, tr=2.0)
    rng_a = s.high - s.low
    range_over_sqrt_a = rng_a / math.sqrt(len(s))
    assert s.atr_mean == pytest.approx(2.0)
    assert range_over_sqrt_a == pytest.approx(2.0 / 6.0)  # 36 -> sqrt 6
    assert not math.isclose(s.atr_mean, range_over_sqrt_a)

    # Case B: a clean trend. Per-bar TR is tiny (~step+2*wick) but the session
    # range is huge, so range/sqrt(n) is far above the true mean TR.
    step = 1.0
    wick = 0.1
    closes = list(np.arange(100.0, 100.0 + 60 * step, step))  # 60 bars trending up
    s2 = S.session_from_closes(closes, wick=wick)
    rng_b = s2.high - s2.low
    range_over_sqrt_b = rng_b / math.sqrt(len(s2))
    # Mean TR for this tape: each bar moves `step`, plus a `wick` on each side.
    # It must be near step + 2*wick and FAR below range/sqrt(n).
    assert s2.atr_mean == pytest.approx(step + 2 * wick, abs=0.05)
    assert range_over_sqrt_b > 3.0 * s2.atr_mean
    assert not math.isclose(s2.atr_mean, range_over_sqrt_b)


# ----------------------------------------------------------------------------
# 5. opening_range, running extremes, high_time/low_time
# ----------------------------------------------------------------------------
def test_opening_range_window_is_first_n_minutes():
    """opening_range(m) covers exactly bars with datetime < open + m minutes."""
    closes = [100, 105, 103, 108, 102, 110, 99, 104, 106, 101]
    s = S.session_from_closes(closes)
    b = s.bars
    # First 3 bars are 09:30, 09:31, 09:32 (datetime < 09:33).
    cutoff = s.open_time + pd.Timedelta(minutes=3)
    win = b[b["datetime"] < cutoff]
    assert len(win) == 3
    expected = (float(win["high"].max()), float(win["low"].min()))
    assert s.opening_range(3) == expected
    # Full-day opening range (huge window) equals the whole-session extremes.
    assert s.opening_range(10_000) == (s.high, s.low)


def test_running_high_low_are_causal_and_monotone():
    """running_high is a non-decreasing cummax; running_low non-increasing
    cummin. Both are causal (corrupting the future leaves the past untouched)."""
    closes = [100, 105, 103, 108, 102, 110, 99, 104, 106, 101]
    df = S.bars_from_closes(closes)
    s = Session.from_intraday("X", df)
    rh = s.running_high.to_numpy()
    rl = s.running_low.to_numpy()
    # Monotonicity.
    assert np.all(np.diff(rh) >= -1e-12)
    assert np.all(np.diff(rl) <= 1e-12)
    # Equal to cummax/cummin of the bar highs/lows.
    np.testing.assert_allclose(rh, s.bars["high"].cummax().to_numpy())
    np.testing.assert_allclose(rl, s.bars["low"].cummin().to_numpy())
    # Causality: corrupt bars after k, the past running extremes are unchanged.
    k = 5
    s2 = Session.from_intraday("X", S.corrupt_future(df, k))
    np.testing.assert_allclose(
        s.running_high.iloc[: k + 1].to_numpy(),
        s2.running_high.iloc[: k + 1].to_numpy(),
    )
    np.testing.assert_allclose(
        s.running_low.iloc[: k + 1].to_numpy(),
        s2.running_low.iloc[: k + 1].to_numpy(),
    )


def test_high_time_and_low_time_point_at_the_extreme_bars():
    """high_time/low_time identify the bar where the session extreme occurs."""
    closes = [100, 105, 103, 108, 102, 110, 99, 104, 106, 101]
    s = S.session_from_closes(closes)
    b = s.bars
    # The bar carrying the session high must actually reach s.high there.
    hi_rows = b[b["datetime"] == s.high_time]
    lo_rows = b[b["datetime"] == s.low_time]
    assert len(hi_rows) == 1 and len(lo_rows) == 1
    assert float(hi_rows["high"].iloc[0]) == pytest.approx(s.high)
    assert float(lo_rows["low"].iloc[0]) == pytest.approx(s.low)
    # No earlier/later bar exceeds the extreme at its own time.
    assert b["high"].max() == pytest.approx(s.high)
    assert b["low"].min() == pytest.approx(s.low)


def test_minutes_from_open_starts_at_zero_and_increments():
    """minutes_from_open is 0 at the open and matches the 1-min bar spacing."""
    s = S.session_from_closes([100, 101, 102, 103, 104, 105])
    mfo = s.minutes_from_open.to_numpy()
    assert mfo[0] == pytest.approx(0.0)
    np.testing.assert_allclose(mfo, np.arange(len(s), dtype=float))
