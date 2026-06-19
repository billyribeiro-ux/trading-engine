"""
Signal generation: turn a VALIDATED model into actionable trades.

This is the layer that converts honest out-of-sample edge into entries. The
non-negotiable rule (HARD RULE #5): a signal may only be emitted from a config
that SURVIVED walk-forward validation, net of costs, over baseline, and after
multiple-testing correction across the whole screen. "No signal" is a correct,
expected output. A scanner that always finds setups is broken.

Two modes:

  * batch_rank  (the morning screen): across a watchlist, for each symbol and
    direction, build the labeled frame, walk-forward-validate it, FDR-correct
    across the ENTIRE screen, keep the survivors, fit a final model on the full
    lookback, then score the CURRENT session's events and emit ranked signals.

  * evaluate_live (the right edge): given an already-validated config, re-dissect
    the developing session from CLOSED BARS ONLY (the forming bar is dropped,
    reusing intraday.live.drop_forming_bar), score the latest event, and fire one
    signal iff the model says go. Never acts on the forming bar.

Horizon-agnostic by construction (see the three-scanner plan): the core
batch_rank/emit logic depends only on a `frame_builder`, a `current_provider`
yielding ScorableEvents, and a bracket -- NOT on Session/Timeframe directly. The
intraday wiring lives in `intraday_config`; a swing or portfolio scanner supplies
its own builder/provider and reuses everything downstream. This also makes the
generator unit-testable offline with no network (inject synthetic providers).

Signal geometry uses the SAME bracket and the SAME causal ATR the labels used, so
entry/stop/target are exactly the trade the validation measured -- never a
different bracket than the one that earned the edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.client import FMPClient
from ..intraday.bars import Timeframe, fetch_intraday
from ..intraday.live import drop_forming_bar
from ..session.dissect import dissect_session
from ..session.pivots import decompose
from ..session.session import build_sessions
from .dataset import build_training_frame, feature_columns
from .features import _causal_atr, extract_session_events
from .labels import REVERSAL_BRACKET, BracketSpec, brackets_for_timeframe
from .model import SignalModel, make_model
from .validate import ValidationReport, fdr_correct_reports, walk_forward_validate

# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """An actionable, validation-backed trade.

    entry/stop/target are PRICES (bracket x causal ATR off the event price). The
    validation fields explain WHY the signal exists -- a user (or the dashboard)
    sees the OOS edge, FDR-corrected p, AUC, and decay backing every entry.
    """

    symbol: str
    timestamp: pd.Timestamp
    direction: str  # 'long' | 'short'
    event_type: str
    entry: float
    stop: float
    target: float
    atr: float
    probability: float
    # --- validation backing (the "why") ---
    oos_edge_r: float
    p_value_fdr: float
    oos_auc: float
    decay: float
    n_events: int
    n_signals: int
    bracket_name: str
    reward_risk: float
    proba_threshold: float
    max_bars: int = 0  # bracket holding horizon (bars); used by the forward journal

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_share(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr(self) -> float:
        risk = self.risk_per_share
        return self.reward_per_share / risk if risk > 0 else float("nan")

    def as_dict(self) -> dict[str, object]:
        """Plain-types view for the API / dashboard (timestamp -> ISO string)."""
        return {
            "symbol": self.symbol,
            "timestamp": pd.Timestamp(self.timestamp).isoformat(),
            "direction": self.direction,
            "event_type": self.event_type,
            "entry": round(self.entry, 4),
            "stop": round(self.stop, 4),
            "target": round(self.target, 4),
            "atr": round(self.atr, 4),
            "rr": round(self.rr, 3),
            "probability": round(self.probability, 4),
            "oos_edge_r": round(self.oos_edge_r, 4),
            "p_value_fdr": round(self.p_value_fdr, 4),
            "oos_auc": round(self.oos_auc, 4),
            "decay": round(self.decay, 4),
            "n_events": self.n_events,
            "n_signals": self.n_signals,
            "bracket": self.bracket_name,
            "max_bars": self.max_bars,
        }


@dataclass(frozen=True)
class ScorableEvent:
    """A current-period event ready to score: the minimal, horizon-agnostic unit.

    `features` is keyed WITHOUT the `f_` prefix (as EventFeatures.features). `atr`
    is the CAUSAL ATR at the event bar -- the same normalizer the labels used.
    """

    event_type: str
    timestamp: pd.Timestamp
    price: float
    atr: float
    features: dict[str, float]


@dataclass(frozen=True)
class ValidatedConfig:
    """A trained + validated config that may legitimately emit signals."""

    symbol: str
    direction: str
    model: SignalModel
    feature_cols: tuple[str, ...]
    report: ValidationReport
    bracket: BracketSpec
    proba_threshold: float


@dataclass(frozen=True)
class ScreenResult:
    """Output of a batch screen: emitted signals + full transparency."""

    signals: tuple[Signal, ...]  # ranked, validated-only
    survivors: tuple[ValidatedConfig, ...]  # configs that passed the gate
    reports: tuple[ValidationReport, ...]  # EVERY config evaluated, FDR-corrected


@dataclass(frozen=True)
class ScannerConfig:
    """Everything the generator needs, decoupled from the data source.

    frame_builder(symbol)    -> labeled training frame (f_* features, y_* labels,
                                a 'y_direction' column).
    current_provider(symbol) -> list[ScorableEvent] for the CURRENT period.
    bracket                  -> the bracket used for BOTH labels and signal
                                geometry (they must match).
    The thresholds are the survival gate (HARD RULE #5).
    """

    frame_builder: Callable[[str], pd.DataFrame]
    current_provider: Callable[[str], Sequence[ScorableEvent]]
    bracket: BracketSpec
    directions: tuple[str, ...] = ("long", "short")
    proba_threshold: float = 0.55
    n_folds: int = 5
    fdr: float = 0.10
    min_edge_r: float = 0.0
    max_decay: float = 0.75
    min_events: int = 40
    cost_r: float = 0.05
    # Minimum OOS signals a config must have taken to be trustworthy. A big edge
    # on a handful of trades is the multiple-testing tail, not skill (e.g. a real
    # screen surfaced AMZN at +1.23R / p_fdr=0.014 on just 8 signals with AUC
    # 0.568 -- near chance). Don't emit from <min_signals OOS trades.
    min_signals: int = 20


# ---------------------------------------------------------------------------
# Geometry + scoring (pure, hand-verifiable)
# ---------------------------------------------------------------------------


def bracket_levels(
    entry: float, atr: float, bracket: BracketSpec, direction: str
) -> tuple[float, float]:
    """(stop, target) prices for a bracket off `entry`, normalized by `atr`.

    Identical to the simulation in labels._bracket_outcome, so the emitted trade
    is exactly the one the validation measured. Long risks down / targets up;
    short mirrors.
    """
    if direction == "long":
        return entry - bracket.stop_atr * atr, entry + bracket.target_atr * atr
    if direction == "short":
        return entry + bracket.stop_atr * atr, entry - bracket.target_atr * atr
    raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")


def _feature_vector(feature_cols: Sequence[str], ev: ScorableEvent) -> np.ndarray:
    """Build the model input row from an event, aligned to feature_cols.

    feature_cols are `f_`-prefixed; ScorableEvent.features are not. A feature the
    event doesn't carry becomes NaN and is mean-imputed by the model's scaler.
    """
    row = [ev.features.get(c[2:] if c.startswith("f_") else c, np.nan) for c in feature_cols]
    return np.array([row], dtype=float)


def _score(model: SignalModel, feature_cols: Sequence[str], ev: ScorableEvent) -> float:
    return float(model.predict_proba(_feature_vector(feature_cols, ev))[0])


# ---------------------------------------------------------------------------
# Batch rank (the morning screen)
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    symbol: str
    direction: str
    frame: pd.DataFrame
    feature_cols: tuple[str, ...]
    report: ValidationReport


def _validate_candidates(symbols: Sequence[str], config: ScannerConfig) -> list[_Candidate]:
    """Per (symbol, direction): build the frame, split by direction, validate.

    Validation is PER DIRECTION because the feature vector is identical for the
    long and short rows of one event -- only the label differs. Mixing them would
    ask one model to predict opposite outcomes from identical inputs. A long model
    predicts P(long bracket wins); a short model P(short bracket wins).
    """
    out: list[_Candidate] = []
    for sym in symbols:
        df = config.frame_builder(sym)
        if df is None or df.empty:
            continue
        fc = tuple(feature_columns(df))
        if not fc:
            continue
        for direction in config.directions:
            if "y_direction" in df.columns:
                sub = df[df["y_direction"] == direction].reset_index(drop=True)
            else:
                sub = df
            if len(sub) < config.min_events:
                continue
            report = walk_forward_validate(
                sub,
                list(fc),
                label_col="y_win",
                r_col="y_bracket_r",
                n_folds=config.n_folds,
                proba_threshold=config.proba_threshold,
                horizon_bars=config.bracket.max_bars,
                cost_r=config.cost_r,
            )
            out.append(_Candidate(sym, direction, sub, fc, report))
    return out


def _survives(report: ValidationReport, config: ScannerConfig) -> bool:
    """The survival gate. Every clause is necessary; none alone is sufficient."""
    return (
        report.n_total_signals >= config.min_signals
        and report.p_value_fdr < config.fdr
        and report.oos_net_expectancy_r > config.min_edge_r
        and report.decay <= config.max_decay
    )


def batch_rank(symbols: Sequence[str], config: ScannerConfig) -> ScreenResult:
    """Screen a watchlist and emit validated, ranked current-session signals.

    Pipeline: validate per (symbol, direction) -> FDR-correct across the WHOLE
    screen (HARD RULE #4) -> keep survivors (HARD RULE #5) -> fit a final model on
    the full lookback -> score the CURRENT events -> emit ranked signals. Returns
    every evaluated report too, so "nothing survived" is visible and auditable.
    """
    candidates = _validate_candidates(symbols, config)
    if not candidates:
        return ScreenResult((), (), ())

    # Multiple-testing correction across EVERY config tried in this screen.
    corrected = fdr_correct_reports([c.report for c in candidates])
    for cand, rep in zip(candidates, corrected):
        cand.report = rep

    survivors: list[ValidatedConfig] = []
    for cand in candidates:
        if not _survives(cand.report, config):
            continue
        model = make_model("logistic", names=list(cand.feature_cols))
        model.fit(
            cand.frame[list(cand.feature_cols)].to_numpy(dtype=float),
            cand.frame["y_win"].to_numpy(dtype=int),
        )
        survivors.append(
            ValidatedConfig(
                symbol=cand.symbol,
                direction=cand.direction,
                model=model,
                feature_cols=cand.feature_cols,
                report=cand.report,
                bracket=config.bracket,
                proba_threshold=config.proba_threshold,
            )
        )

    # Score the current period once per symbol (shared across its directions).
    current: dict[str, Sequence[ScorableEvent]] = {}
    signals: list[Signal] = []
    for vc in survivors:
        if vc.symbol not in current:
            current[vc.symbol] = list(config.current_provider(vc.symbol))
        for ev in current[vc.symbol]:
            if not (np.isfinite(ev.atr) and ev.atr > 0 and np.isfinite(ev.price)):
                continue
            p = _score(vc.model, vc.feature_cols, ev)
            if not (np.isfinite(p) and p >= config.proba_threshold):
                continue
            stop, target = bracket_levels(ev.price, ev.atr, vc.bracket, vc.direction)
            signals.append(_build_signal(vc, ev, p, stop, target, config.proba_threshold))

    signals.sort(key=lambda s: (s.oos_edge_r, s.probability), reverse=True)
    return ScreenResult(tuple(signals), tuple(survivors), tuple(c.report for c in candidates))


def _build_signal(
    vc: ValidatedConfig,
    ev: ScorableEvent,
    probability: float,
    stop: float,
    target: float,
    proba_threshold: float,
) -> Signal:
    r = vc.report
    return Signal(
        symbol=vc.symbol,
        timestamp=pd.Timestamp(ev.timestamp),
        direction=vc.direction,
        event_type=ev.event_type,
        entry=float(ev.price),
        stop=float(stop),
        target=float(target),
        atr=float(ev.atr),
        probability=float(probability),
        oos_edge_r=r.oos_net_expectancy_r,
        p_value_fdr=r.p_value_fdr,
        oos_auc=r.oos_auc,
        decay=r.decay,
        n_events=r.n_events,
        n_signals=r.n_total_signals,
        bracket_name=vc.bracket.name,
        reward_risk=vc.bracket.reward_risk,
        proba_threshold=proba_threshold,
        max_bars=vc.bracket.max_bars,
    )


# ---------------------------------------------------------------------------
# Live (the right edge of the developing session)
# ---------------------------------------------------------------------------


def evaluate_live(
    client: FMPClient,
    validated: ValidatedConfig,
    timeframe: Timeframe = Timeframe.M5,
    now: pd.Timestamp | None = None,
    history_days: int = 3,
) -> Signal | None:
    """Score the LATEST closed-bar event of the developing session.

    Reuses intraday.live.drop_forming_bar: the in-progress bar is removed before
    any structure is read, so a signal can never repaint when the bar closes
    (HARD RULE: never act on the forming bar). Returns one Signal for the most
    recent event iff the validated model clears the threshold, else None.

    `validated` must come from a prior batch_rank / validation run -- we do NOT
    re-validate every bar; the gate was already passed.
    """
    events = developing_session_events(
        client, validated.symbol, timeframe, now=now, history_days=history_days
    )
    if not events:
        return None
    ev = events[-1]  # the latest decision bar (events are sorted by index)
    if not (np.isfinite(ev.atr) and ev.atr > 0 and np.isfinite(ev.price)):
        return None
    p = _score(validated.model, validated.feature_cols, ev)
    if not (np.isfinite(p) and p >= validated.proba_threshold):
        return None
    stop, target = bracket_levels(ev.price, ev.atr, validated.bracket, validated.direction)
    return _build_signal(validated, ev, p, stop, target, validated.proba_threshold)


def developing_session_events(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe = Timeframe.M5,
    now: pd.Timestamp | None = None,
    history_days: int = 3,
) -> list[ScorableEvent]:
    """The current session's events from CLOSED BARS ONLY (forming bar dropped).

    Fetches recent intraday, drops the in-progress bar, builds the latest session,
    dissects it, and returns ScorableEvents. Shared by live evaluation and any
    right-edge UI. Empty list if the session is too thin to dissect.
    """
    symbol = symbol.strip().upper()
    to_date = (now or pd.Timestamp.now()).normalize()
    from_date = to_date - pd.Timedelta(days=history_days)
    bars = fetch_intraday(
        client,
        symbol,
        timeframe,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=(to_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if bars.empty:
        return []
    bar_minutes = Timeframe(timeframe).minutes
    bars = drop_forming_bar(bars, now=now, bar_minutes=bar_minutes)
    sessions = build_sessions(bars, symbol)
    if not sessions:
        return []
    session = sessions[-1]
    dissection = dissect_session(session, decomposition=decompose(session))
    out: list[ScorableEvent] = []
    for ev in extract_session_events(session, dissection):
        out.append(
            ScorableEvent(
                event_type=ev.event_type,
                timestamp=ev.event_time,
                price=float(ev.event_price),
                atr=float(_causal_atr(session, ev.event_index)),
                features=dict(ev.features),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Intraday wiring + rendering
# ---------------------------------------------------------------------------


def intraday_config(
    client: FMPClient,
    timeframe: Timeframe = Timeframe.M5,
    lookback_days: int = 60,
    style: str = "reversal",
    **overrides: object,
) -> ScannerConfig:
    """A ScannerConfig wired to the intraday data layer.

    The SAME bracket drives build_training_frame's labels and the signal geometry.
    `overrides` tune the survival gate (proba_threshold, fdr, min_edge_r, ...).
    """
    minutes = Timeframe(timeframe).minutes
    bracket = brackets_for_timeframe(minutes).get(style) or REVERSAL_BRACKET

    def frame_builder(sym: str) -> pd.DataFrame:
        return build_training_frame(
            client, sym, timeframe, lookback_days=lookback_days, style=style
        )

    def current_provider(sym: str) -> list[ScorableEvent]:
        return developing_session_events(client, sym, timeframe)

    return ScannerConfig(
        frame_builder=frame_builder,
        current_provider=current_provider,
        bracket=bracket,
        **overrides,  # type: ignore[arg-type]
    )


def render_signals(result: ScreenResult) -> str:
    """A compact, desk-grade table of the screen -- honest about empties."""
    lines = [
        f"Screen: {len(result.reports)} configs evaluated, "
        f"{len(result.survivors)} survived, {len(result.signals)} signals."
    ]
    if not result.signals:
        lines.append("  (no validated signals — no significant edge survived the screen)")
        return "\n".join(lines)
    lines.append(
        f"  {'SYMBOL':<8}{'DIR':<6}{'EVENT':<16}{'ENTRY':>9}{'STOP':>9}"
        f"{'TARGET':>9}{'R:R':>6}{'PROB':>7}{'EDGE_R':>8}{'p_fdr':>8}"
    )
    for s in result.signals:
        lines.append(
            f"  {s.symbol:<8}{s.direction:<6}{s.event_type:<16}"
            f"{s.entry:>9.2f}{s.stop:>9.2f}{s.target:>9.2f}{s.rr:>6.2f}"
            f"{s.probability:>7.2f}{s.oos_edge_r:>+8.3f}{s.p_value_fdr:>8.3f}"
        )
    return "\n".join(lines)
