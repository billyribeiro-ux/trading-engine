"""
Swing scanner config — plugs straight into the shared batch_rank pipeline.

Demonstrates the three-scanner architecture: NO new validation, FDR, signal, or
ranking code. `swing_config` supplies a daily frame_builder + a current-swings
provider + a daily bracket; `engine.ml.signals.batch_rank` does the rest exactly
as it does for intraday.

    from engine.ml.signals import batch_rank
    from engine.swing.scanner import swing_config
    res = batch_rank(["AAPL", "MSFT"], swing_config(client))
"""

from __future__ import annotations

from ..data.client import FMPClient
from ..ml.features import _causal_atr
from ..ml.signals import ScannerConfig, ScorableEvent
from .dataset import SWING_BRACKET, build_swing_frame, build_swing_window
from .features import DEFAULT_SWING_SCALE, extract_swing_events


def swing_config(
    client: FMPClient,
    lookback_days: int = 730,
    scale_atr: float = DEFAULT_SWING_SCALE,
    recent_bars: int = 10,
    **overrides: object,
) -> ScannerConfig:
    """A ScannerConfig wired to the daily-bar swing layer.

    `recent_bars`: only swing legs confirmed within the last `recent_bars` days
    are emitted as CURRENT signals (an old swing isn't actionable today). Gate
    defaults are looser than intraday because a multi-year daily window yields
    fewer events than a 60-day 5-min window.
    """

    def frame_builder(sym: str):
        return build_swing_frame(client, sym, lookback_days, SWING_BRACKET, scale_atr)

    def current_provider(sym: str) -> list[ScorableEvent]:
        window = build_swing_window(client, sym, lookback_days)
        if window is None:
            return []
        n = len(window)
        out: list[ScorableEvent] = []
        for ev in extract_swing_events(window, scale_atr):
            if ev.event_index < n - recent_bars:  # only recent, actionable swings
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
        "min_events": 20,
        "n_folds": 4,
        "min_signals": 8,
        "max_decay": 1.0,
        "cost_r": 0.05,
    }
    defaults.update(overrides)
    return ScannerConfig(
        frame_builder=frame_builder,
        current_provider=current_provider,
        bracket=SWING_BRACKET,
        **defaults,  # type: ignore[arg-type]
    )
