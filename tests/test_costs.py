"""
Adversarial tests for engine.intraday.costs.

Every expected value here is hand-computed from the documented formulas, not
read back from the implementation. The model is supposed to be *conservative*
(friction is a real cost, stops are costlier than targets), so the assertions
encode the direction of the bias as well as the magnitude.

Hand-computation reference (default CostModel):
    commission_per_share = 0.005   commission_per_trade = 0.0
    min_commission       = 1.0     half_spread_bps      = 1.0
    slippage_atr_frac    = 0.05    entry_is_marketable  = True

    half_spread(price) = price * 1.0 / 1e4 = price * 1e-4
    slip(atr)          = atr * 0.05
"""

from __future__ import annotations

import pytest

from engine.intraday.costs import FRICTIONLESS, CostModel, Side


# ---------------------------------------------------------------------------
# 1. entry_fill
# ---------------------------------------------------------------------------
def test_entry_fill_long_fills_above_signal_by_half_spread_plus_slip():
    cm = CostModel()  # all defaults
    signal, atr = 100.0, 2.0
    half_spread = signal * 1.0 / 1e4  # 0.01
    slip = atr * 0.05  # 0.10
    expected = signal + half_spread + slip  # 100.11
    assert cm.entry_fill(Side.LONG, signal, atr) == pytest.approx(expected)
    # LONG entry is strictly WORSE (higher) than the signal.
    assert cm.entry_fill(Side.LONG, signal, atr) > signal


def test_entry_fill_short_fills_below_signal_by_same_amount():
    cm = CostModel()
    signal, atr = 100.0, 2.0
    half_spread = signal * 1.0 / 1e4  # 0.01
    slip = atr * 0.05  # 0.10
    expected = signal - half_spread - slip  # 99.89
    assert cm.entry_fill(Side.SHORT, signal, atr) == pytest.approx(expected)
    # SHORT entry is strictly WORSE (lower) than the signal.
    assert cm.entry_fill(Side.SHORT, signal, atr) < signal


def test_entry_fill_long_and_short_are_symmetric_about_signal():
    cm = CostModel()
    signal, atr = 100.0, 2.0
    long_gap = cm.entry_fill(Side.LONG, signal, atr) - signal
    short_gap = signal - cm.entry_fill(Side.SHORT, signal, atr)
    assert long_gap == pytest.approx(short_gap)
    assert long_gap == pytest.approx(0.11)


def test_entry_fill_non_marketable_drops_the_slippage_term_only():
    cm = CostModel(entry_is_marketable=False)
    signal, atr = 100.0, 2.0
    half_spread = signal * 1.0 / 1e4  # 0.01
    # No slippage when entry is a passive limit; half-spread still paid.
    assert cm.entry_fill(Side.LONG, signal, atr) == pytest.approx(signal + half_spread)
    assert cm.entry_fill(Side.SHORT, signal, atr) == pytest.approx(signal - half_spread)
    # And the slip term really was the difference vs the marketable model.
    cm_mkt = CostModel(entry_is_marketable=True)
    slip = atr * 0.05
    assert (
        cm_mkt.entry_fill(Side.LONG, signal, atr) - cm.entry_fill(Side.LONG, signal, atr)
    ) == pytest.approx(slip)


# ---------------------------------------------------------------------------
# 2. stop_fill
# ---------------------------------------------------------------------------
def test_stop_fill_long_is_below_the_trigger_by_half_spread_plus_slip():
    cm = CostModel()
    stop, atr = 95.0, 2.0
    half_spread = stop * 1.0 / 1e4  # 0.0095
    slip = atr * 0.05  # 0.10
    expected = stop - half_spread - slip  # 94.8905
    assert cm.stop_fill(Side.LONG, stop, atr) == pytest.approx(expected)
    # A touched LONG stop fills WORSE (below) the trigger.
    assert cm.stop_fill(Side.LONG, stop, atr) < stop


def test_stop_fill_short_is_above_the_trigger_by_half_spread_plus_slip():
    cm = CostModel()
    stop, atr = 105.0, 2.0
    half_spread = stop * 1.0 / 1e4  # 0.0105
    slip = atr * 0.05  # 0.10
    expected = stop + half_spread + slip  # 105.1105
    assert cm.stop_fill(Side.SHORT, stop, atr) == pytest.approx(expected)
    # A touched SHORT stop fills WORSE (above) the trigger.
    assert cm.stop_fill(Side.SHORT, stop, atr) > stop


# ---------------------------------------------------------------------------
# 3. target_fill -- resting limit, half-spread but NO slippage
# ---------------------------------------------------------------------------
def test_target_fill_long_pays_half_spread_no_slippage():
    cm = CostModel()
    target = 105.0
    half_spread = target * 1.0 / 1e4  # 0.0105
    expected = target - half_spread  # 104.9895
    assert cm.target_fill(Side.LONG, target) == pytest.approx(expected)
    # Slippage (atr*frac = 0.10 for atr=2) must NOT be in the target fill.
    # If it were, the LONG target fill would be 104.8795, not 104.9895.
    assert cm.target_fill(Side.LONG, target) != pytest.approx(target - half_spread - 0.10)


def test_target_fill_short_pays_half_spread_no_slippage():
    cm = CostModel()
    target = 95.0
    half_spread = target * 1.0 / 1e4  # 0.0095
    expected = target + half_spread  # 95.0095
    assert cm.target_fill(Side.SHORT, target) == pytest.approx(expected)
    assert cm.target_fill(Side.SHORT, target) != pytest.approx(target + half_spread + 0.10)


# ---------------------------------------------------------------------------
# 4. round_trip_pnl -- LONG and SHORT, per-share + per-trade commission
# ---------------------------------------------------------------------------
def test_round_trip_long_target_per_share_commission():
    cm = CostModel()
    r = cm.round_trip_pnl(
        Side.LONG,
        entry_signal=100.0,
        exit_price=105.0,
        atr=2.0,
        shares=100.0,
        exit_is_stop=False,
    )
    # eff_entry = 100 + 0.01 + 0.10           = 100.11
    # eff_exit  = target_fill(LONG, 105)      = 104.9895
    # gross_per_share = 105 - 100 = 5 -> gross_pnl = 500
    # net_per_share   = 104.9895 - 100.11 = 4.8795 -> *100 = 487.95
    # commission      = max(1.0, 100*0.005)=1.0 per fill -> 2.0 total
    # net_pnl         = 487.95 - 2.0 = 485.95
    # spread_slippage = gross_pnl - net_per_share*shares = 500 - 487.95 = 12.05
    assert r["eff_entry"] == pytest.approx(100.11)
    assert r["eff_exit"] == pytest.approx(104.9895)
    assert r["gross_pnl"] == pytest.approx(500.0)
    assert r["commission"] == pytest.approx(2.0)
    assert r["net_pnl"] == pytest.approx(485.95)
    assert r["net_pnl_per_share"] == pytest.approx(4.8595)
    assert r["spread_slippage_cost"] == pytest.approx(12.05)
    # Friction is a real cost: net is strictly below gross.
    assert r["net_pnl"] < r["gross_pnl"]


def test_round_trip_short_target_per_share_commission():
    cm = CostModel()
    r = cm.round_trip_pnl(
        Side.SHORT,
        entry_signal=100.0,
        exit_price=95.0,
        atr=2.0,
        shares=100.0,
        exit_is_stop=False,
    )
    # eff_entry = entry_fill(SHORT,100,2) = 99.89
    # eff_exit  = target_fill(SHORT,95)   = 95.0095
    # gross_per_share = 100 - 95 = 5 -> gross_pnl = 500
    # net_per_share   = eff_entry - eff_exit = 99.89 - 95.0095 = 4.8805 -> *100 = 488.05
    # commission      = 2.0 ; net_pnl = 486.05
    # spread_slippage = 500 - 488.05 = 11.95
    assert r["eff_entry"] == pytest.approx(99.89)
    assert r["eff_exit"] == pytest.approx(95.0095)
    assert r["gross_pnl"] == pytest.approx(500.0)
    assert r["commission"] == pytest.approx(2.0)
    assert r["net_pnl"] == pytest.approx(486.05)
    assert r["spread_slippage_cost"] == pytest.approx(11.95)
    assert r["net_pnl"] < r["gross_pnl"]


def test_round_trip_per_trade_commission_is_flat_not_scaled():
    cm = CostModel(
        commission_per_share=0.0,
        commission_per_trade=1.0,
        min_commission=0.0,
    )
    r = cm.round_trip_pnl(
        Side.LONG,
        entry_signal=100.0,
        exit_price=105.0,
        atr=2.0,
        shares=1000.0,
        exit_is_stop=False,
    )
    # Per-trade commission is flat: 1.0 entry + 1.0 exit = 2.0 regardless of size.
    assert r["commission"] == pytest.approx(2.0)
    # net_per_share = 104.9895 - 100.11 = 4.8795 -> *1000 = 4879.5 ; -2.0 = 4877.5
    assert r["net_pnl"] == pytest.approx(4877.5)
    assert r["gross_pnl"] == pytest.approx(5000.0)
    assert r["spread_slippage_cost"] == pytest.approx(120.5)


def test_commission_min_floor_vs_above_floor():
    cm = CostModel()  # per_share=0.005, min=1.0
    # 50 shares -> 50*0.005 = 0.25 < 1.0 floor -> 1.0 per fill -> 2.0 total
    r_floor = cm.round_trip_pnl(
        Side.LONG,
        100.0,
        105.0,
        2.0,
        shares=50.0,
        exit_is_stop=False,
    )
    assert r_floor["commission"] == pytest.approx(2.0)
    # 400 shares -> 400*0.005 = 2.0 > 1.0 floor -> 2.0 per fill -> 4.0 total
    r_above = cm.round_trip_pnl(
        Side.LONG,
        100.0,
        105.0,
        2.0,
        shares=400.0,
        exit_is_stop=False,
    )
    assert r_above["commission"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 5. FRICTIONLESS -- gross == net, the explicit gross/net baseline
# ---------------------------------------------------------------------------
def test_frictionless_gross_equals_net():
    r = FRICTIONLESS.round_trip_pnl(
        Side.LONG,
        entry_signal=100.0,
        exit_price=105.0,
        atr=2.0,
        shares=100.0,
        exit_is_stop=False,
    )
    assert r["gross_pnl"] == pytest.approx(500.0)
    assert r["net_pnl"] == pytest.approx(r["gross_pnl"])
    assert r["commission"] == pytest.approx(0.0)
    assert r["spread_slippage_cost"] == pytest.approx(0.0)
    # eff prices equal the raw signal/exit -- no spread, no slippage.
    assert r["eff_entry"] == pytest.approx(100.0)
    assert r["eff_exit"] == pytest.approx(105.0)


def test_frictionless_short_gross_equals_net():
    r = FRICTIONLESS.round_trip_pnl(
        Side.SHORT,
        entry_signal=100.0,
        exit_price=95.0,
        atr=2.0,
        shares=100.0,
        exit_is_stop=True,  # stop exit still frictionless
    )
    assert r["gross_pnl"] == pytest.approx(500.0)
    assert r["net_pnl"] == pytest.approx(500.0)
    assert r["spread_slippage_cost"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Conservative direction: stop exit costlier than target exit, same move
# ---------------------------------------------------------------------------
def test_stop_exit_is_costlier_than_target_exit_for_same_price_move():
    cm = CostModel()
    # SAME exit price (105) and same everything; only exit_is_stop differs.
    target_exit = cm.round_trip_pnl(
        Side.LONG,
        100.0,
        105.0,
        2.0,
        shares=100.0,
        exit_is_stop=False,
    )
    stop_exit = cm.round_trip_pnl(
        Side.LONG,
        100.0,
        105.0,
        2.0,
        shares=100.0,
        exit_is_stop=True,
    )
    # Stop exit applies slippage on the exit; target exit does not.
    # target net = 485.95 ; stop net:
    #   eff_exit = stop_fill(LONG,105,2) = 105 - 0.0105 - 0.10 = 104.8895
    #   net_per_share = 104.8895 - 100.11 = 4.7795 -> *100 = 477.95 ; -2.0 = 475.95
    assert target_exit["net_pnl"] == pytest.approx(485.95)
    assert stop_exit["net_pnl"] == pytest.approx(475.95)
    assert stop_exit["net_pnl"] < target_exit["net_pnl"]
    # The cost gap is exactly the slippage on one leg: atr*frac*shares = 0.10*100 = 10.0
    assert (target_exit["net_pnl"] - stop_exit["net_pnl"]) == pytest.approx(10.0)
