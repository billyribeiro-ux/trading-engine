"""
Thin FastAPI service over the trading engine — the dashboard backend.

Wraps the existing Python engine (no logic here): /screen runs the validated
batch-rank scanner, /dissect returns one session's structured dissection,
/capabilities reports the detected FMP tier. The heavy lifting lives in
engine.ml.signals and engine.session.runner; this layer only marshals JSON.

Testability: the engine operations are FastAPI dependencies (get_screener /
get_dissector / get_client). Tests override them with fakes, so the suite runs
fully offline with zero network. This mirrors the scanner-agnostic seam used in
signals.py — the same idea, at the HTTP boundary.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..data.client import FMPClient
from ..export import to_bytes
from ..forward.journal import SignalJournal
from ..forward.live import _DEFAULT_JOURNAL
from ..intraday.bars import Timeframe
from ..ml.signals import ScreenResult, batch_rank, intraday_config
from ..portfolio.scanner import portfolio_config
from ..session.runner import dissect_real_session
from ..settings import SettingsStore, mask_key, resolve_api_key
from ..swing.scanner import swing_config
from .serialize import dissection_to_dict, screen_to_dict

# A screener maps a request -> ScreenResult; a dissector maps (symbol, tf, date)
# -> (session, dissection, nested). Both are injected so tests can fake them.
Screener = Callable[["ScreenRequest"], ScreenResult]
Dissector = Callable[[str, Timeframe, Optional[str]], tuple]
KeyValidator = Callable[[str], dict]


_SCANNER_DEFAULT_LOOKBACK = {"intraday": 60, "swing": 730, "portfolio": 2920}

# Default live universe for in-app scan (the validated swing-long names + peers).
_DEFAULT_LIVE_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "KO",
    "PEP",
    "DIS",
    "CSCO",
    "INTC",
    "ORCL",
    "CRM",
    "ADBE",
    "COST",
    "HD",
]


class ScreenRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Watchlist tickers")
    scanner: str = Field("intraday", description="'intraday' | 'swing' | 'portfolio'")
    timeframe: str = Field("5min", description="Intraday bar resolution (intraday only)")
    lookback_days: Optional[int] = Field(
        None, ge=5, le=3650, description="The lookback knob; None -> per-scanner default"
    )
    style: str = Field("reversal", description="Intraday style: 'reversal' or 'scalp'")
    proba_threshold: float = Field(0.55, ge=0.5, lt=1.0)
    fdr: float = Field(0.10, gt=0.0, le=0.5)
    min_edge_r: float = Field(0.0, description="Minimum OOS edge over baseline (R)")


class SettingsBody(BaseModel):
    fmp_api_key: str = Field(..., min_length=8, description="Financial Modeling Prep API key")


class JournalActionBody(BaseModel):
    symbols: Optional[list[str]] = Field(None, description="Universe; None -> default live set")
    scanner: str = Field("swing", description="'intraday' | 'swing' | 'portfolio'")
    model: str = Field("gbt", description="'logistic' | 'gbt'")
    direction: str = Field("long", description="'long' | 'short'")


class ExportBody(BaseModel):
    format: str = Field("csv", pattern="^(csv|xlsx)$")
    filename: str = Field("export", min_length=1, max_length=80)
    sheets: dict[str, list[dict]] = Field(..., description="{sheet_name: [row, ...]}")


@lru_cache(maxsize=4)
def _client_for(key: str) -> FMPClient:
    return FMPClient(key)


def _resolve_store() -> SettingsStore:
    path = os.environ.get("SETTINGS_PATH")
    return SettingsStore(Path(path) if path else None)


def get_settings_store() -> SettingsStore:
    return _resolve_store()


def get_key_validator() -> KeyValidator:
    """Validate a key by asking FMP what it can do (overridden in tests)."""

    def validate(key: str) -> dict:
        return _client_for(key).capabilities()

    return validate


def get_client() -> FMPClient:
    key = resolve_api_key(_resolve_store())  # saved key wins, FMP_API_KEY env fallback
    if not key:
        raise HTTPException(
            status_code=503, detail="FMP API key not configured — add it in Settings"
        )
    return _client_for(key)


def get_screener(client: FMPClient = Depends(get_client)) -> Screener:
    """Default screener dispatching to the chosen scanner. Overridden in tests.

    One pipeline (batch_rank), three configs — the only thing that varies is which
    ScannerConfig is built (intraday / swing / portfolio)."""

    def run(req: ScreenRequest) -> ScreenResult:
        gate = {
            "proba_threshold": req.proba_threshold,
            "fdr": req.fdr,
            "min_edge_r": req.min_edge_r,
        }
        lookback = req.lookback_days or _SCANNER_DEFAULT_LOOKBACK.get(req.scanner, 60)
        symbols = [s.strip().upper() for s in req.symbols]
        if req.scanner == "swing":
            cfg = swing_config(client, lookback_days=lookback, **gate)
        elif req.scanner == "portfolio":
            cfg = portfolio_config(client, lookback_days=lookback, **gate)
        else:
            cfg = intraday_config(
                client,
                timeframe=Timeframe(req.timeframe),
                lookback_days=lookback,
                style=req.style,
                **gate,
            )
        return batch_rank(symbols, cfg)

    return run


def get_journal() -> SignalJournal:
    """The live forward-test journal. JOURNAL_PATH overrides the default."""
    path = os.environ.get("JOURNAL_PATH")
    return SignalJournal(Path(path) if path else _DEFAULT_JOURNAL)


def get_dissector(client: FMPClient = Depends(get_client)) -> Dissector:
    """Default dissector wired to the engine runner. Overridden in tests."""

    def run(symbol: str, timeframe: Timeframe, on_date: str | None):
        return dissect_real_session(client, symbol, timeframe, on_date=on_date)

    return run


def get_live_scan() -> Callable:
    """The live scan op (fit validated model, score current events, log). Overridden
    in tests so the endpoint wiring is verified offline."""
    from ..forward.live import scan_and_log

    return scan_and_log


def get_live_resolve() -> Callable:
    """The live resolve op (resolve open journal entries vs fresh bars)."""
    from ..forward.live import resolve

    return resolve


def _journal_payload(j: SignalJournal) -> dict:
    s = j.summary()
    return {
        "open": s.n_open,
        "resolved": s.n_resolved,
        "realized_mean_r": round(s.realized_mean_r, 4),
        "realized_hit_rate": round(s.realized_hit_rate, 4),
        "validated_edge_r": round(s.mean_validated_edge_r, 4),
        "by_reason": s.by_reason,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Trading Engine API", version="0.1.0")
    # Allow ANY localhost port — Vite drifts (5173 -> 5174 -> ...) when a port is
    # taken, and the browser blocks cross-origin calls if that port isn't allowed.
    # Scoped to localhost/127.0.0.1 only (local dev app), not external origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities(client: FMPClient = Depends(get_client)) -> dict:
        return client.capabilities()

    @app.get("/journal")
    def journal(j: SignalJournal = Depends(get_journal)) -> dict:
        """Live forward-test journal: realized-vs-validated summary + recent entries.
        Needs no FMP key (reads the local journal file)."""
        return {"summary": _journal_payload(j), "entries": j.entries()[-200:]}

    @app.post("/journal/scan")
    def journal_scan(
        body: JournalActionBody,
        client: FMPClient = Depends(get_client),
        j: SignalJournal = Depends(get_journal),
        scan=Depends(get_live_scan),
    ) -> dict:
        """Run a live scan and append fired signals to the journal (in-app)."""
        from ..forward.live import LiveConfig

        syms = tuple(s.strip().upper() for s in (body.symbols or _DEFAULT_LIVE_UNIVERSE))
        live = LiveConfig(
            symbols=syms,
            scanner=body.scanner,
            model_kind=body.model,
            direction=body.direction,
            journal_path=j.path,
        )
        sigs = scan(client, live, j)
        return {
            "logged": len(sigs),
            "signals": [s.as_dict() for s in sigs[:50]],
            "summary": _journal_payload(j),
        }

    @app.post("/journal/resolve")
    def journal_resolve(
        body: JournalActionBody,
        client: FMPClient = Depends(get_client),
        j: SignalJournal = Depends(get_journal),
        resolver=Depends(get_live_resolve),
    ) -> dict:
        """Resolve open journal entries against fresh bars (in-app)."""
        from ..forward.live import LiveConfig

        live = LiveConfig(symbols=(), scanner=body.scanner, journal_path=j.path)
        rows = resolver(client, live, j)
        resolved = sum(1 for r in rows if r.get("status") == "resolved")
        return {"resolved": resolved, "total": len(rows), "summary": _journal_payload(j)}

    @app.post("/export")
    def export_results(body: ExportBody) -> Response:
        """Serialize result rows to a CSV or multi-sheet XLSX download. No FMP key
        needed — the browser posts the rows it already has."""
        import pandas as pd

        frames = {name: pd.DataFrame(rows) for name, rows in body.sheets.items()}
        data = to_bytes(frames, body.format)
        ext = "xlsx" if body.format == "xlsx" else "csv"
        media = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if body.format == "xlsx"
            else "text/csv"
        )
        return Response(
            content=data,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{body.filename}.{ext}"'},
        )

    @app.get("/settings")
    def read_settings(store: SettingsStore = Depends(get_settings_store)) -> dict:
        """Dashboard settings status. Never returns the full key — only a mask."""
        saved = store.get_api_key()
        env = os.environ.get("FMP_API_KEY")
        return {
            "configured": bool(saved or env),
            "source": "saved" if saved else ("env" if env else None),
            "masked": store.masked_key() if saved else (mask_key(env) if env else None),
        }

    @app.post("/settings")
    def write_settings(
        body: SettingsBody,
        store: SettingsStore = Depends(get_settings_store),
        validate: KeyValidator = Depends(get_key_validator),
    ) -> dict:
        """Validate the FMP key against the API, then persist it (0600). A saved key
        takes precedence over the FMP_API_KEY env var on the next request."""
        key = body.fmp_api_key.strip()
        try:
            caps = validate(key)
        except Exception as exc:  # bad key, network, tier probe failure
            raise HTTPException(status_code=400, detail=f"FMP rejected the key: {exc}") from exc
        store.set_api_key(key)
        return {"ok": True, "masked": store.masked_key(), "tier": caps.get("tier")}

    @app.post("/screen")
    def screen(req: ScreenRequest, screener: Screener = Depends(get_screener)) -> dict:
        if req.scanner not in _SCANNER_DEFAULT_LOOKBACK:
            raise HTTPException(status_code=422, detail=f"unknown scanner: {req.scanner}")
        if req.scanner == "intraday":
            try:
                Timeframe(req.timeframe)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"bad timeframe: {req.timeframe}"
                ) from exc
        result = screener(req)
        return screen_to_dict(result)

    @app.get("/dissect/{symbol}")
    def dissect(
        symbol: str,
        timeframe: str = "5min",
        date: Optional[str] = None,
        dissector: Dissector = Depends(get_dissector),
    ) -> dict:
        try:
            tf = Timeframe(timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"bad timeframe: {timeframe}") from exc
        try:
            session, dissection, nested = dissector(symbol.strip().upper(), tf, date)
        except ValueError as exc:  # no data / no session for the request
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return dissection_to_dict(session, dissection, nested)

    return app


app = create_app()
