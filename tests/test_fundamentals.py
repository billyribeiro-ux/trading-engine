"""
Fundamental AS-OF causality (Scanner #3 deep layer) + survivorship inclusion.

The crown jewel: a fundamental filed AFTER a decision date can NEVER influence the
features at that date. The income statement's `date` is the fiscal period end, but
it is only known at `acceptedDate` (weeks later) — using it earlier is the worst
lookahead. These tests construct reports with explicit as-of dates and prove the
join is causal, plus that survivorship inclusion flags delisted names.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.core.universe import build_universe, classify
from engine.portfolio.fundamentals import FundamentalSeries, _Earnings, _Income


def _series_with(future: bool) -> FundamentalSeries:
    # 5 consecutive known quarters (all filed before the decision date 2026-03-01).
    income = [
        _Income(pd.Timestamp("2025-01-15"), eps=1.0, revenue=100.0),
        _Income(pd.Timestamp("2025-04-15"), eps=1.1, revenue=110.0),
        _Income(pd.Timestamp("2025-07-15"), eps=1.2, revenue=120.0),
        _Income(pd.Timestamp("2025-10-15"), eps=1.3, revenue=130.0),
        _Income(pd.Timestamp("2026-01-15"), eps=1.5, revenue=150.0),
    ]
    if future:  # a blockbuster report filed AFTER the decision date — must be ignored
        income.append(_Income(pd.Timestamp("2026-05-01"), eps=99.0, revenue=99999.0))
    earnings = [
        _Earnings(
            pd.Timestamp("2026-02-15"), eps_actual=2.0, eps_est=1.6, rev_actual=0.0, rev_est=0.0
        )
    ]
    return FundamentalSeries("X", income=sorted(income, key=lambda r: r.asof), earnings=earnings)


DECISION = pd.Timestamp("2026-03-01")


def test_future_filing_never_affects_a_past_decision():
    """The exclusion of a report filed after the decision date IS the guarantee."""
    with_future = _series_with(future=True).asof(DECISION)
    without_future = _series_with(future=False).asof(DECISION)
    assert with_future == without_future
    # And the blockbuster's numbers do NOT leak in.
    assert with_future["eps_yoy"] == pytest.approx(0.5)  # (1.5 - 1.0) / 1.0
    assert with_future["rev_yoy"] == pytest.approx(0.5)  # (150 - 100) / 100


def test_eps_surprise_from_latest_known_earnings():
    feats = _series_with(future=False).asof(DECISION)
    assert feats["eps_surprise"] == pytest.approx((2.0 - 1.6) / 1.6)  # 0.25


def test_asof_before_a_filing_excludes_it():
    """asof one day BEFORE the latest filing must fall back to the prior quarter."""
    s = _series_with(future=False)
    just_before = s.asof(pd.Timestamp("2026-01-14"))  # before the 2026-01-15 filing
    # Latest known is the 2025-10-15 report (eps 1.3); YoY needs 5 -> only 4 known
    # here, so eps_yoy is absent, and days_since_report points at the Oct filing.
    assert "eps_yoy" not in just_before
    assert just_before["fund_days_since_report"] == float(
        (pd.Timestamp("2026-01-14") - pd.Timestamp("2025-10-15")).days
    )


def test_no_fundamentals_known_yet_returns_empty():
    assert _series_with(future=False).asof(pd.Timestamp("2024-01-01")) == {}


def test_build_universe_classifies_and_flags():
    members = build_universe(["AAPL", "SQQQ"])
    assert len(members) == 2
    by = {m.symbol: m for m in members}
    assert not by["AAPL"].is_derived
    assert by["SQQQ"].is_derived and by["SQQQ"].inverse_of == "QQQ"
    assert classify("AAPL").symbol == "AAPL"


# --- real data (gated) -----------------------------------------------------


@pytest.mark.realdata
@pytest.mark.skipif(not os.environ.get("FMP_API_KEY"), reason="no FMP_API_KEY")
def test_real_fundamentals_are_asof_causal_and_no_leak():
    from engine.data.client import FMPClient
    from engine.ml.validate import _auc
    from engine.portfolio.dataset import build_position_frame
    from engine.portfolio.window import build_weekly_window

    client = FMPClient(os.environ["FMP_API_KEY"])
    fs = FundamentalSeries.from_client(client, "AAPL")
    assert fs.income, "expected income statements"
    # As-of one day before the latest filing must NOT include that filing.
    latest = fs.income[-1].asof
    before = fs.asof(latest - pd.Timedelta(days=1))
    after = fs.asof(latest)
    assert before.get("fund_days_since_report", 0) > after.get("fund_days_since_report", -1)

    # The position frame carries fundamental features and has no perfect predictor.
    bench = build_weekly_window(client, "SPY")
    df = build_position_frame(client, "AAPL", benchmark=bench, use_fundamentals=True)
    if df.empty:
        pytest.skip("no position frame")
    assert any(c.startswith("f_eps") or c.startswith("f_rev") for c in df.columns), (
        "no fundamental features"
    )
    fcols = [c for c in df.columns if c.startswith("f_")]
    worst = 0.0
    for direction in ("long", "short"):
        sub = df[df["y_direction"] == direction]
        y = sub["y_win"].to_numpy(dtype=int)
        if y.sum() in (0, len(y)):
            continue
        for col in fcols:
            x = sub[col].to_numpy(dtype=float)
            m = np.isfinite(x)
            if m.sum() < 25:
                continue
            worst = max(worst, abs(_auc(y[m], x[m]) - 0.5))
    assert worst < 0.49, f"near-PERFECT predictor (leak?): |AUC-0.5|={worst:.3f}"


@pytest.mark.realdata
@pytest.mark.skipif(not os.environ.get("FMP_API_KEY"), reason="no FMP_API_KEY")
def test_fetch_delisted_flags_members():
    from engine.core.universe import fetch_delisted
    from engine.data.client import FMPClient

    members = fetch_delisted(FMPClient(os.environ["FMP_API_KEY"]), limit=20)
    if not members:
        pytest.skip("no delisted list returned")
    assert all(m.delisted for m in members)
    assert any(m.delisted_date for m in members)
