"""engine.settings: store round-trip, masking, 0600 perms, and key resolution
(saved key wins over the FMP_API_KEY env, shared by the API and every CLI)."""

from __future__ import annotations

import os
import stat

from engine.settings import SettingsStore, mask_key, resolve_api_key


def test_mask_key_shows_only_last_four():
    assert mask_key("ABCDEFGH1234").endswith("1234")
    assert "ABCDEFGH" not in mask_key("ABCDEFGH1234")
    assert mask_key("ab") == "••"
    assert mask_key("") == ""


def test_store_roundtrip_trims_and_is_0600(tmp_path):
    s = SettingsStore(tmp_path / "s.json")
    assert s.get_api_key() is None
    s.set_api_key("  mykey123456  ")
    assert s.get_api_key() == "mykey123456"  # trimmed
    assert s.masked_key().endswith("3456")
    assert stat.S_IMODE(os.stat(s.path).st_mode) == 0o600  # secret-tight perms


def test_resolve_prefers_saved_over_env(tmp_path, monkeypatch):
    s = SettingsStore(tmp_path / "s.json")
    monkeypatch.setenv("FMP_API_KEY", "envkey123")
    assert resolve_api_key(s) == "envkey123"  # nothing saved -> env fallback
    s.set_api_key("savedkey123")
    assert resolve_api_key(s) == "savedkey123"  # saved key wins
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    assert resolve_api_key(s) == "savedkey123"


def test_resolve_none_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    assert resolve_api_key(SettingsStore(tmp_path / "s.json")) is None


def test_store_survives_corrupt_file(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not valid json")
    s = SettingsStore(p)
    assert s.load() == {} and s.get_api_key() is None
    s.set_api_key("recover12345")  # overwrites the corrupt file
    assert s.get_api_key() == "recover12345"
