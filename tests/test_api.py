"""
API layer tests — fully offline via dependency overrides (no FMP, no network).

The engine operations (screener, dissector, client) are FastAPI dependencies, so
we swap in fakes and assert the HTTP marshalling: the screen response shape, the
honest-empty case, the structured dissection (incl. the STRUCTURE==LEG ROLES
invariant carried through to JSON), tier capabilities, and error handling.
"""

from __future__ import annotations

import _synth as S
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from engine.api.app import (
    app,
    get_client,
    get_dissector,
    get_journal,
    get_key_validator,
    get_screener,
    get_settings_store,
)
from engine.ml.signals import ScreenResult, Signal
from engine.ml.validate import ValidationReport
from engine.session.dissect import dissect_session
from engine.session.nested import build_nested_structure
from engine.session.pivots import decompose


def _signal() -> Signal:
    # entry 100, atr 2, 2:1 long bracket -> stop 98, target 104 (hand-computed).
    return Signal(
        symbol="TSLA",
        timestamp=pd.Timestamp("2026-06-18 10:00"),
        direction="long",
        event_type="vwap_reclaim",
        entry=100.0,
        stop=98.0,
        target=104.0,
        atr=2.0,
        probability=0.71,
        oos_edge_r=0.42,
        p_value_fdr=0.03,
        oos_auc=0.66,
        decay=0.05,
        n_events=220,
        n_signals=18,
        bracket_name="reversal",
        reward_risk=2.0,
        proba_threshold=0.55,
    )


def _report(symbol: str, p_fdr: float) -> ValidationReport:
    return ValidationReport(
        symbol=symbol,
        n_events=220,
        n_folds=5,
        oos_net_expectancy_r=0.42,
        oos_hit_rate=0.55,
        oos_auc=0.66,
        p_value=0.02,
        p_value_fdr=p_fdr,
        folds=(),
        decay=0.05,
        n_total_signals=18,
    )


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _override_screener(result: ScreenResult):
    app.dependency_overrides[get_screener] = lambda: lambda req: result


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_screen_returns_ranked_signals(client):
    _override_screener(ScreenResult((_signal(),), (), (_report("TSLA", 0.03),)))
    r = client.post("/screen", json={"symbols": ["TSLA"], "lookback_days": 60})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n_signals"] == 1
    sig = body["signals"][0]
    assert sig["symbol"] == "TSLA" and sig["direction"] == "long"
    assert (sig["entry"], sig["stop"], sig["target"]) == (100.0, 98.0, 104.0)
    assert sig["p_value_fdr"] == 0.03
    assert body["reports"][0]["symbol"] == "TSLA"


def test_screen_honest_empty(client):
    """No survivors -> zero signals, reported plainly (not an error)."""
    _override_screener(ScreenResult((), (), (_report("TSLA", 0.40),)))
    r = client.post("/screen", json={"symbols": ["TSLA", "NVDA"]})
    assert r.status_code == 200
    assert r.json()["summary"]["n_signals"] == 0
    assert r.json()["signals"] == []


def test_screen_rejects_bad_timeframe(client):
    _override_screener(ScreenResult((), (), ()))
    r = client.post("/screen", json={"symbols": ["TSLA"], "timeframe": "7min"})
    assert r.status_code == 422


def test_screen_requires_symbols(client):
    r = client.post("/screen", json={"symbols": []})
    assert r.status_code == 422  # pydantic min_length


def test_screen_accepts_all_three_scanners(client):
    _override_screener(ScreenResult((_signal(),), (), (_report("TSLA", 0.03),)))
    for scanner in ("intraday", "swing", "portfolio"):
        r = client.post("/screen", json={"symbols": ["TSLA"], "scanner": scanner})
        assert r.status_code == 200, scanner
        assert r.json()["summary"]["n_signals"] == 1


def test_screen_rejects_unknown_scanner(client):
    _override_screener(ScreenResult((), (), ()))
    r = client.post("/screen", json={"symbols": ["TSLA"], "scanner": "quantum"})
    assert r.status_code == 422


def test_swing_timeframe_not_validated(client):
    """timeframe is intraday-only; a swing screen ignores it (no 422)."""
    _override_screener(ScreenResult((), (), ()))
    r = client.post("/screen", json={"symbols": ["TSLA"], "scanner": "swing", "timeframe": "7min"})
    assert r.status_code == 200


def test_dissect_returns_structured_consistent_report(client):
    def fake_dissect(symbol, tf, on_date):
        ses = S.multileg_session(np.random.default_rng(0), n_legs=8)
        dec = decompose(ses)
        dis = dissect_session(ses, decomposition=dec)
        nested = build_nested_structure(dec, dis.scale_atr)
        return ses, dis, nested

    app.dependency_overrides[get_dissector] = lambda: fake_dissect
    r = client.get("/dissect/TSLA?timeframe=5min")
    assert r.status_code == 200
    body = r.json()
    assert "header" in body and "structure" in body and "leg_roles" in body
    # STRUCTURE == LEG ROLES carried through to JSON (bug-#3 invariant).
    assert len(body["structure"]) == len(body["leg_roles"])
    assert body["consistent"] is True
    assert body["header"]["open"] > 0


def test_dissect_404_when_no_session(client):
    def fake_dissect(symbol, tf, on_date):
        raise ValueError("No session for ZZZZ on 2099-01-01")

    app.dependency_overrides[get_dissector] = lambda: fake_dissect
    r = client.get("/dissect/ZZZZ?date=2099-01-01")
    assert r.status_code == 404


def test_journal_endpoint_reports_summary_and_entries(client, tmp_path):
    from engine.forward.journal import SignalJournal

    j = SignalJournal(tmp_path / "j.jsonl")
    j.log([_signal()], scanner="swing")
    app.dependency_overrides[get_journal] = lambda: j
    r = client.get("/journal")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["open"] == 1
    assert body["summary"]["resolved"] == 0
    assert len(body["entries"]) == 1
    assert body["entries"][0]["symbol"] == "TSLA"


def test_settings_get_unconfigured(client, tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    from engine.api.settings import SettingsStore

    app.dependency_overrides[get_settings_store] = lambda: SettingsStore(tmp_path / "s.json")
    r = client.get("/settings")
    assert r.status_code == 200
    assert r.json() == {"configured": False, "source": None, "masked": None}


def test_settings_post_validates_saves_and_masks(client, tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    from engine.api.settings import SettingsStore

    store = SettingsStore(tmp_path / "s.json")
    app.dependency_overrides[get_settings_store] = lambda: store
    app.dependency_overrides[get_key_validator] = lambda: lambda k: {"tier": "ULTIMATE"}
    r = client.post("/settings", json={"fmp_api_key": "ABCDEFGH1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["tier"] == "ULTIMATE"
    assert body["masked"].endswith("1234") and "ABCDEFGH" not in body["masked"]
    assert store.get_api_key() == "ABCDEFGH1234"  # persisted full key
    # GET now reports configured from the saved key
    g = client.get("/settings").json()
    assert g["configured"] and g["source"] == "saved" and g["masked"].endswith("1234")


def test_settings_post_rejects_bad_key(client, tmp_path):
    from engine.api.settings import SettingsStore

    store = SettingsStore(tmp_path / "s.json")
    app.dependency_overrides[get_settings_store] = lambda: store

    def boom(_key):
        raise RuntimeError("401 Unauthorized")

    app.dependency_overrides[get_key_validator] = lambda: boom
    r = client.post("/settings", json={"fmp_api_key": "badkey12345"})
    assert r.status_code == 400
    assert store.get_api_key() is None  # not saved on failure


def test_settings_post_requires_min_length(client):
    r = client.post("/settings", json={"fmp_api_key": "short"})
    assert r.status_code == 422  # pydantic min_length


def test_capabilities_reports_tier(client):
    class FakeClient:
        def capabilities(self):
            return {"tier": "ULTIMATE", "rate_limit_per_min": 3000}

    app.dependency_overrides[get_client] = lambda: FakeClient()
    r = client.get("/capabilities")
    assert r.status_code == 200 and r.json()["tier"] == "ULTIMATE"
