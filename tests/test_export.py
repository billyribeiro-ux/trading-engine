"""
Result export: results round-trip to CSV and multi-sheet Excel intact.
"""

from __future__ import annotations

import pandas as pd

from engine.export import (
    bakeoff_to_df,
    export,
    gauntlet_to_frames,
    journal_to_df,
    signals_to_df,
)
from engine.forward.bakeoff import VersionResult
from engine.forward.gauntlet import GauntletVerdict
from engine.forward.runner import ForwardTestResult, RollingForwardResult
from engine.ml.signals import Signal


def _signal() -> Signal:
    return Signal(
        symbol="TSLA",
        timestamp=pd.Timestamp("2026-06-19 10:00"),
        direction="long",
        event_type="swing_leg",
        entry=100.0,
        stop=98.0,
        target=104.0,
        atr=2.0,
        probability=0.7,
        oos_edge_r=0.5,
        p_value_fdr=0.01,
        oos_auc=0.63,
        decay=0.04,
        n_events=200,
        n_signals=40,
        bracket_name="swing",
        reward_risk=2.0,
        proba_threshold=0.55,
        max_bars=10,
    )


def _ftr(**kw) -> ForwardTestResult:
    base = dict(
        symbol="?",
        n_train=900,
        n_holdout=400,
        n_holdout_signals=60,
        validated_edge_r=0.5,
        validated_p=0.01,
        realized_edge_r=0.55,
        realized_hit_rate=0.5,
        realized_p=0.001,
        holdout_auc=0.63,
        forward_decay_r=-0.05,
        persisted=True,
        n_holdout_days=40,
    )
    base.update(kw)
    return ForwardTestResult(**base)


def _verdict() -> GauntletVerdict:
    rows = (
        VersionResult("gbt", "gbt", _ftr(), 0.001, True),
        VersionResult("logistic", "logistic", _ftr(realized_edge_r=0.46), 0.002, True),
    )
    rolling = {
        "gbt": RollingForwardResult(6, 6, (), 0.60, 0.0, 337, True),
        "logistic": RollingForwardResult(6, 6, (), 0.56, 0.0, 224, True),
    }
    return GauntletVerdict("long", 13543, 1256, rows, rolling, True)


def test_signals_and_bakeoff_to_df():
    sdf = signals_to_df([_signal()])
    assert {"symbol", "entry", "stop", "target", "max_bars"} <= set(sdf.columns)
    assert sdf.iloc[0]["symbol"] == "TSLA"
    bdf = bakeoff_to_df(_verdict().bakeoff)
    assert {"version", "model", "realized_edge_r", "promoted"} <= set(bdf.columns)
    assert len(bdf) == 2


def test_export_excel_multisheet_roundtrip(tmp_path):
    out = export(gauntlet_to_frames(_verdict()), tmp_path / "gauntlet.xlsx")
    assert out == [tmp_path / "gauntlet.xlsx"] and out[0].exists()
    book = pd.read_excel(out[0], sheet_name=None)  # all sheets
    assert {"summary", "bakeoff", "rolling"} <= set(book)
    assert book["summary"].iloc[0]["passed"]
    assert len(book["bakeoff"]) == 2


def test_export_csv_single_and_multi(tmp_path):
    one = export({"signals": signals_to_df([_signal()])}, tmp_path / "sig.csv")
    assert one == [tmp_path / "sig.csv"]
    assert pd.read_csv(one[0]).iloc[0]["symbol"] == "TSLA"

    many = export(gauntlet_to_frames(_verdict()), tmp_path / "g.csv")
    names = {p.name for p in many}
    assert names == {"g_summary.csv", "g_bakeoff.csv", "g_rolling.csv"}
    assert all(p.exists() for p in many)


def test_journal_to_df():
    df = journal_to_df(
        [{"symbol": "AAPL", "status": "open"}, {"symbol": "MSFT", "status": "resolved"}]
    )
    assert list(df["symbol"]) == ["AAPL", "MSFT"]
