"""
Pooled cross-symbol training: the power single names lack.

The forward tests proved the bottleneck is signal/power, not model — single names
yield only a handful of holdout signals. Pooling events across the whole universe
into ONE time-sorted frame gives train and holdout enough events to detect a small
REAL edge (if one exists) and to reject noise with confidence. It reuses each
scanner's `frame_builder` (the ScannerConfig seam), so the same code pools
intraday, swing, or portfolio events. Everything downstream — forward_test,
bake_off, FDR — is unchanged; only the data is broader.

The pooled frame is sorted by (date, event_index) so the holdout split is a
genuine out-of-time split across the universe: the model trains on the past of all
names and is scored on the future of all names.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from ..ml.dataset import feature_columns
from ..ml.signals import ScannerConfig
from .bakeoff import ModelVersion, VersionResult, bake_off

logger = logging.getLogger("engine.forward.pooled")


def build_pooled_frame(
    symbols: Sequence[str], config: ScannerConfig, direction: str | None = None
) -> pd.DataFrame:
    """Concatenate per-symbol labeled frames into one time-ordered pool.

    Uses `config.frame_builder` so the pool matches whatever scanner the config
    is for. A symbol that errors or has no events is skipped (logged), so one bad
    name never aborts the pool. Optionally filter to a single direction.
    """
    frames: list[pd.DataFrame] = []
    for s in symbols:
        sym = s.strip().upper()
        try:
            df = config.frame_builder(sym)
        except Exception as exc:  # one bad symbol shouldn't kill the pool
            logger.warning("pooled: skipped %s: %s", sym, exc)
            continue
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    if direction and "y_direction" in pooled.columns:
        pooled = pooled[pooled["y_direction"] == direction]
    return pooled.sort_values(["date", "event_index"]).reset_index(drop=True)


def pooled_bakeoff(
    symbols: Sequence[str],
    config: ScannerConfig,
    versions: Sequence[ModelVersion] | None = None,
    *,
    direction: str = "long",
    cutoff_frac: float = 0.7,
    fdr: float = 0.10,
    min_holdout_signals: int = 20,
    min_holdout_days: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[VersionResult]]:
    """Build the pooled frame for one direction and bake versions off against the
    forward gate. Returns (pooled_frame, version_results)."""
    pooled = build_pooled_frame(symbols, config, direction=direction)
    if pooled.empty:
        return pooled, []
    rows = bake_off(
        pooled,
        feature_columns(pooled),
        versions,
        horizon_bars=config.bracket.max_bars,
        cutoff_frac=cutoff_frac,
        cost_r=config.cost_r,
        fdr=fdr,
        min_holdout_signals=min_holdout_signals,
        min_holdout_days=min_holdout_days,
        seed=seed,
    )
    return pooled, rows
