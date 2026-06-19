"""
Live forward testing: turn the historically-validated edge into LIVE evidence.

The gauntlet (pooling -> day-guard -> FDR -> rolling -> fresh symbols) proved an
edge OUT-OF-TIME on history. The only thing that converts that into trust is LIVE
accumulation: log today's signals, resolve them as real bars print, and compare
realized to the validated edge. This module does exactly that, reusing the
ScannerConfig seam so it works for any scanner.

Discipline preserved end-to-end:
  * The live model is the SAME validated config, fit on the pooled universe.
  * Signals are emitted ONLY if the pooled config STILL passes the forward gate
    now (persisted) — if the edge has decayed, nothing fires.
  * Each Signal carries the current forward-test backing (edge/p/auc) as its "why".
  * Resolution is closed-bars-only, conservative stop-first (no lookahead).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..data.client import FMPClient
from ..ml.dataset import feature_columns
from ..ml.model import make_model
from ..ml.signals import ScannerConfig, Signal, _score, bracket_levels
from .journal import SignalJournal
from .pooled import build_pooled_frame
from .runner import forward_test

logger = logging.getLogger("engine.forward.live")

_DEFAULT_JOURNAL = Path.home() / ".cache" / "fmp_engine" / "signal_journal.jsonl"


@dataclass(frozen=True)
class LiveConfig:
    symbols: tuple[str, ...]
    scanner: str = "swing"  # 'intraday' | 'swing' | 'portfolio'
    model_kind: str = "gbt"
    direction: str = "long"
    proba_threshold: float = 0.55
    lookback_days: int = 2920
    journal_path: Path = field(default_factory=lambda: _DEFAULT_JOURNAL)


def _config_for(client: FMPClient, live: LiveConfig) -> ScannerConfig:
    # local imports to avoid importing every scanner unless used
    if live.scanner == "swing":
        from ..swing.scanner import swing_config

        return swing_config(client, lookback_days=live.lookback_days)
    if live.scanner == "portfolio":
        from ..portfolio.scanner import portfolio_config

        return portfolio_config(client, lookback_days=live.lookback_days)
    from ..intraday.bars import Timeframe
    from ..ml.signals import intraday_config

    return intraday_config(client, timeframe=Timeframe.M5, lookback_days=live.lookback_days)


def generate_live_signals(
    symbols: Sequence[str],
    config: ScannerConfig,
    *,
    model_kind: str = "gbt",
    direction: str = "long",
    proba_threshold: float = 0.55,
    min_holdout_days: int = 10,
) -> tuple[list[Signal], object]:
    """Fit the validated model on the pooled universe and emit CURRENT signals.

    Emits nothing unless the pooled config STILL passes the forward gate now
    (persisted) — live emission requires the edge to be alive today. Returns
    (signals, forward_test_result_used_as_backing)."""
    pooled = build_pooled_frame(symbols, config, direction=direction)
    if pooled.empty:
        return [], None
    fc = feature_columns(pooled)
    bt = forward_test(
        pooled,
        fc,
        horizon_bars=config.bracket.max_bars,
        cost_r=config.cost_r,
        model_kind=model_kind,
        min_holdout_signals=20,
        min_holdout_days=min_holdout_days,
    )
    if not bt.persisted:
        logger.warning(
            "live: pooled %s config does NOT currently pass the forward gate "
            "(edge=%+.3f p=%.3f days=%d) -- emitting NO signals.",
            direction,
            bt.realized_edge_r,
            bt.realized_p,
            bt.n_holdout_days,
        )
        return [], bt

    model = make_model(model_kind, names=fc)
    model.fit(pooled[fc].to_numpy(dtype=float), pooled["y_win"].to_numpy(dtype=int))

    signals: list[Signal] = []
    for sym in symbols:
        sym = sym.strip().upper()
        for ev in config.current_provider(sym):
            if not (ev.atr == ev.atr and ev.atr > 0 and ev.price == ev.price):
                continue
            p = _score(model, fc, ev)
            if not (p == p and p >= proba_threshold):
                continue
            stop, target = bracket_levels(ev.price, ev.atr, config.bracket, direction)
            signals.append(
                Signal(
                    symbol=sym,
                    timestamp=pd.Timestamp(ev.timestamp),
                    direction=direction,
                    event_type=ev.event_type,
                    entry=float(ev.price),
                    stop=float(stop),
                    target=float(target),
                    atr=float(ev.atr),
                    probability=float(p),
                    oos_edge_r=bt.realized_edge_r,
                    p_value_fdr=bt.realized_p,
                    oos_auc=bt.holdout_auc,
                    decay=bt.forward_decay_r,
                    n_events=bt.n_holdout,
                    n_signals=bt.n_holdout_signals,
                    bracket_name=config.bracket.name,
                    reward_risk=config.bracket.reward_risk,
                    proba_threshold=proba_threshold,
                    max_bars=config.bracket.max_bars,
                )
            )
    signals.sort(key=lambda s: s.probability, reverse=True)
    return signals, bt


def scan_and_log(client: FMPClient, live: LiveConfig, journal: SignalJournal) -> list[Signal]:
    """Run the live scan and append fired signals to the journal."""
    config = _config_for(client, live)
    signals, _bt = generate_live_signals(
        live.symbols,
        config,
        model_kind=live.model_kind,
        direction=live.direction,
        proba_threshold=live.proba_threshold,
    )
    if signals:
        journal.log(signals, scanner=live.scanner)
    return signals


def resolve(client: FMPClient, live: LiveConfig, journal: SignalJournal) -> list[dict]:
    """Resolve open journal entries against fresh bars (daily for swing/portfolio,
    intraday for intraday)."""
    open_syms = {e["symbol"] for e in journal.entries() if e.get("status") != "resolved"}
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in open_syms:
        try:
            if live.scanner == "intraday":
                from ..intraday.bars import Timeframe, fetch_intraday

                df = fetch_intraday(client, sym, Timeframe.M5)
            else:
                df = client.fetch("eod_full", symbol=sym)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["datetime"] = pd.to_datetime(df["date"])
        except Exception as exc:
            logger.warning("resolve: no bars for %s: %s", sym, exc)
            continue
        if df is not None and not df.empty:
            bars_by_symbol[sym] = df
    return journal.resolve_all(bars_by_symbol)
