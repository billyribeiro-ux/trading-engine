"""
Persisted dashboard settings — the FMP API key today, more later.

Stored as JSON at ~/.config/fmp_engine/settings.json (outside the repo, like the
runtime caches), with 0600 permissions since it holds a secret. The full key is
never returned over the API — only a masked form. The store is intentionally a
plain key/value bag so new settings slot in without schema churn.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

_DEFAULT_PATH = Path.home() / ".config" / "fmp_engine" / "settings.json"


def mask_key(key: str) -> str:
    """Show only the last 4 chars; the rest as dots. Never expose a full secret."""
    k = (key or "").strip()
    if len(k) <= 4:
        return "•" * len(k)
    return "•" * (len(k) - 4) + k[-4:]


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = {**self.load(), **data}  # merge so future settings persist alongside
        self.path.write_text(json.dumps(merged, indent=2))
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — it holds a secret
        except OSError:
            pass

    def get_api_key(self) -> str | None:
        k = self.load().get("fmp_api_key")
        return k.strip() if isinstance(k, str) and k.strip() else None

    def set_api_key(self, key: str) -> None:
        self.save({"fmp_api_key": key.strip()})

    def masked_key(self) -> str | None:
        k = self.get_api_key()
        return mask_key(k) if k else None
