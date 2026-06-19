"""
Forward testing: does a validated edge actually persist out-of-time?

The crown jewel is decay detection — an edge that validates strongly on the train
slice but DIES on the unseen holdout must be flagged `persisted=False`. That is
the whole point of forward testing: catching overfit edges before they're traded.
Plus: a genuinely persistent edge is confirmed, noise stays null, and the live
journal resolves bracket outcomes correctly (conservative stop-first).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.forward.journal import SignalJournal
from engine.forward.runner import forward_test
from engine.ml.signals import Signal

FEATS = ["f_signal", "f_n1", "f_n2"]
N = 900


def _frame(seed: int, *, train_signal: bool, holdout_signal: bool, cut_frac: float = 0.7):
    rng = np.random.default_rng(seed)
    cut = int(N * cut_frac)
    f_signal = np.empty(N)
    y_win = np.empty(N, dtype=int)
    for i in range(N):
        on = train_signal if i < cut else holdout_signal
        latent = rng.normal()
        if on:
            prob = 1.0 / (1.0 + np.exp(-2.5 * latent))
            y_win[i] = rng.binomial(1, prob)
            f_signal[i] = latent + rng.normal(0.0, 0.5)
        else:
            y_win[i] = rng.integers(0, 2)
            f_signal[i] = rng.normal()
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(N), unit="D"),
            "event_index": np.arange(N),
            "f_signal": f_signal,
            "f_n1": rng.normal(size=N),
            "f_n2": rng.normal(size=N),
            "y_win": y_win,
            "y_bracket_r": np.where(y_win == 1, 2.0, -1.0).astype(float),
        }
    )


def test_persistent_edge_is_confirmed():
    res = forward_test(_frame(1, train_signal=True, holdout_signal=True), FEATS, horizon_bars=2)
    assert res.validated_edge_r > 0
    assert res.realized_edge_r > 0
    assert res.realized_p < 0.10
    assert res.persisted


def test_noise_does_not_persist():
    res = forward_test(_frame(2, train_signal=False, holdout_signal=False), FEATS, horizon_bars=2)
    assert not res.persisted
    assert abs(res.realized_edge_r) < 0.25


def test_decayed_edge_is_caught():
    """Strong in train, GONE in holdout -> validates but must NOT be marked
    persisted, with a large forward decay. This is the overfit catcher."""
    res = forward_test(_frame(3, train_signal=True, holdout_signal=False), FEATS, horizon_bars=2)
    assert res.validated_edge_r > 0.10, "train edge should look real"
    assert not res.persisted, "a decayed edge must not survive forward testing"
    assert res.forward_decay_r > 0.20, res.forward_decay_r


def test_few_holdout_days_blocks_promotion():
    """A real, persistent edge measured over too FEW distinct calendar days must
    NOT be promoted — same-day signals are correlated, so a handful of days isn't
    independent evidence (the pooled-intraday 3-day trap)."""
    df = _frame(1, train_signal=True, holdout_signal=True).copy()
    # Collapse to ~9 distinct days so the holdout spans only a few.
    df["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(len(df)) // 100, unit="D")
    res = forward_test(df, FEATS, horizon_bars=2)
    assert res.realized_edge_r > 0  # the edge itself is real
    assert res.n_holdout_days < 10
    assert not res.persisted  # but too few independent days -> not promoted


def _signal(**kw) -> Signal:
    base = dict(
        symbol="TSLA",
        timestamp=pd.Timestamp("2026-06-18 10:00"),
        direction="long",
        event_type="vwap_reclaim",
        entry=100.0,
        stop=98.0,
        target=104.0,
        atr=2.0,
        probability=0.7,
        oos_edge_r=0.4,
        p_value_fdr=0.03,
        oos_auc=0.66,
        decay=0.05,
        n_events=200,
        n_signals=20,
        bracket_name="reversal",
        reward_risk=2.0,
        proba_threshold=0.55,
        max_bars=5,
    )
    base.update(kw)
    return Signal(**base)


def _bars(rows):
    ts = pd.date_range("2026-06-18 10:01", periods=len(rows), freq="1min")
    return pd.DataFrame(
        {
            "datetime": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1000.0,
        }
    )


def test_journal_resolves_target(tmp_path):
    j = SignalJournal(tmp_path / "j.jsonl")
    assert j.log([_signal()], scanner="intraday") == 1
    # long, target 104: bar 2 prints a high of 105, low stays above the stop.
    bars = _bars([(100, 101, 99.5, 100.5), (100.5, 102, 100, 101), (101, 105, 100.5, 104.2)])
    resolved = j.resolve_all({"TSLA": bars})
    e = resolved[0]
    assert e["status"] == "resolved"
    assert e["exit_reason"] == "target"
    assert e["realized_r"] == pytest.approx(2.0)  # reward/risk
    assert e["bars_held"] == 3
    s = j.summary(cost_r=0.05)
    assert s.n_resolved == 1 and s.n_open == 0
    assert s.realized_mean_r == pytest.approx(2.0 - 0.05)


def test_journal_resolves_stop_first_on_both(tmp_path):
    j = SignalJournal(tmp_path / "j.jsonl")
    j.log([_signal()])
    # One bar spans BOTH target (>=104) and stop (<=98) -> conservative stop.
    bars = _bars([(100, 105, 97.0, 100)])
    e = j.resolve_all({"TSLA": bars})[0]
    assert e["exit_reason"] == "stop"
    assert e["realized_r"] == pytest.approx(-1.0)


def test_journal_keeps_open_until_bars_print(tmp_path):
    j = SignalJournal(tmp_path / "j.jsonl")
    j.log([_signal()])
    # Only bars BEFORE the decision time -> nothing to resolve.
    early = _bars([(100, 101, 99, 100)])
    early["datetime"] = pd.date_range("2026-06-18 09:50", periods=1, freq="1min")
    e = j.resolve_all({"TSLA": early})[0]
    assert e["status"] == "open"


@pytest.mark.realdata
@pytest.mark.skipif(not os.environ.get("FMP_API_KEY"), reason="no FMP_API_KEY")
def test_forward_test_runs_on_real_data():
    from engine.data.client import FMPClient
    from engine.intraday.bars import Timeframe
    from engine.ml.dataset import build_training_frame, feature_columns

    client = FMPClient(os.environ["FMP_API_KEY"])
    df = build_training_frame(client, "TSLA", Timeframe.M5, lookback_days=60)
    if len(df) < 100:
        pytest.skip("not enough events")
    long = df[df["y_direction"] == "long"].reset_index(drop=True)
    res = forward_test(long, feature_columns(df), horizon_bars=24)
    # We don't assert an edge exists (likely none); we assert the harness produced
    # a coherent out-of-time verdict.
    assert res.n_train > 0 and res.n_holdout > 0
    assert np.isfinite(res.validated_edge_r) and np.isfinite(res.realized_edge_r)
    assert isinstance(res.persisted, bool)
