"""
Portfolio scanner config — Scanner #3 on the shared batch_rank pipeline.

Like swing_config, supplies a frame_builder + current-positions provider + a
week-sized bracket; everything downstream (validate, FDR, signals, ranking) is
reused unchanged. The benchmark weekly window is fetched ONCE and shared across
the screen so relative-strength features are consistent.

    from engine.ml.signals import batch_rank
    from engine.portfolio.scanner import portfolio_config
    res = batch_rank(["AAPL", "XOM", "JPM"], portfolio_config(client))
"""

from __future__ import annotations

from ..data.client import FMPClient
from ..ml.features import _causal_atr
from ..ml.signals import ScannerConfig, ScorableEvent
from .dataset import POSITION_BRACKET, build_position_frame
from .features import DEFAULT_POSITION_SCALE, extract_position_events
from .window import align_benchmark, build_weekly_window


def portfolio_config(
    client: FMPClient,
    lookback_days: int = 2920,
    benchmark: str = "SPY",
    scale_atr: float = DEFAULT_POSITION_SCALE,
    recent_bars: int = 4,
    **overrides: object,
) -> ScannerConfig:
    """A ScannerConfig wired to the weekly portfolio layer.

    `benchmark` drives relative-strength features (fetched once). `recent_bars`:
    only positions whose leg confirmed within the last few weeks are emitted as
    current signals. Gate defaults are looser still (multi-year weekly windows
    yield the fewest events); cost is small on multi-week holds.
    """
    bench_window = build_weekly_window(client, benchmark, lookback_days)

    def frame_builder(sym: str):
        return build_position_frame(
            client, sym, lookback_days, POSITION_BRACKET, scale_atr, benchmark=bench_window
        )

    def current_provider(sym: str) -> list[ScorableEvent]:
        window = build_weekly_window(client, sym, lookback_days)
        if window is None:
            return []
        bench_close = align_benchmark(window, bench_window) if bench_window is not None else None
        n = len(window)
        out: list[ScorableEvent] = []
        for ev in extract_position_events(window, scale_atr, bench_close):
            if ev.event_index < n - recent_bars:
                continue
            out.append(
                ScorableEvent(
                    event_type=ev.event_type,
                    timestamp=ev.event_time,
                    price=float(ev.event_price),
                    atr=float(_causal_atr(window, ev.event_index)),
                    features=dict(ev.features),
                )
            )
        return out

    defaults: dict[str, object] = {
        "min_events": 12,
        "n_folds": 4,
        "min_signals": 5,
        "max_decay": 1.5,
        "cost_r": 0.02,
    }
    defaults.update(overrides)
    return ScannerConfig(
        frame_builder=frame_builder,
        current_provider=current_provider,
        bracket=POSITION_BRACKET,
        **defaults,  # type: ignore[arg-type]
    )
