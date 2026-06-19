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
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..data.client import FMPClient
from ..intraday.bars import Timeframe
from ..ml.signals import ScreenResult, batch_rank, intraday_config
from ..session.runner import dissect_real_session
from .serialize import dissection_to_dict, screen_to_dict

# A screener maps a request -> ScreenResult; a dissector maps (symbol, tf, date)
# -> (session, dissection, nested). Both are injected so tests can fake them.
Screener = Callable[["ScreenRequest"], ScreenResult]
Dissector = Callable[[str, Timeframe, Optional[str]], tuple]


class ScreenRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, description="Watchlist tickers")
    timeframe: str = Field("5min", description="Bar resolution, e.g. 1min/5min/15min")
    lookback_days: int = Field(60, ge=5, le=365, description="The lookback knob")
    style: str = Field("reversal", description="'reversal' or 'scalp'")
    proba_threshold: float = Field(0.55, ge=0.5, lt=1.0)
    fdr: float = Field(0.10, gt=0.0, le=0.5)
    min_edge_r: float = Field(0.0, description="Minimum OOS edge over baseline (R)")


@lru_cache(maxsize=4)
def _client_for(key: str) -> FMPClient:
    return FMPClient(key)


def get_client() -> FMPClient:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="FMP_API_KEY not set on the server")
    return _client_for(key)


def get_screener(client: FMPClient = Depends(get_client)) -> Screener:
    """Default screener wired to the intraday engine. Overridden in tests."""

    def run(req: ScreenRequest) -> ScreenResult:
        cfg = intraday_config(
            client,
            timeframe=Timeframe(req.timeframe),
            lookback_days=req.lookback_days,
            style=req.style,
            proba_threshold=req.proba_threshold,
            fdr=req.fdr,
            min_edge_r=req.min_edge_r,
        )
        return batch_rank([s.strip().upper() for s in req.symbols], cfg)

    return run


def get_dissector(client: FMPClient = Depends(get_client)) -> Dissector:
    """Default dissector wired to the engine runner. Overridden in tests."""

    def run(symbol: str, timeframe: Timeframe, on_date: str | None):
        return dissect_real_session(client, symbol, timeframe, on_date=on_date)

    return run


def create_app() -> FastAPI:
    app = FastAPI(title="Trading Engine API", version="0.1.0")
    # The SvelteKit dev server runs on 5173; allow local origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities(client: FMPClient = Depends(get_client)) -> dict:
        return client.capabilities()

    @app.post("/screen")
    def screen(req: ScreenRequest, screener: Screener = Depends(get_screener)) -> dict:
        try:
            Timeframe(req.timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"bad timeframe: {req.timeframe}") from exc
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
