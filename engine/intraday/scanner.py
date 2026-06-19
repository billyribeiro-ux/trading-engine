"""
Live reversal scanner across a watchlist, ranked by measured edge.

This ties the two halves together: it runs the historical backtest per (symbol,
scenario) to learn each scenario's real origin-retest hit-rate, then evaluates
the live session and tags every fired/forming/watch signal with that historical
edge. A CONFIRMED signal from a scenario with a proven 70% retest rate and a
tight interval ranks above a CONFIRMED signal from a thin-sample scenario -- so
"show every scenario" stays usable instead of becoming a flat firehose.

A live signal is only as trustworthy as the backtest behind it; that dependency
is enforced here, not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.client import FMPClient
from .backtest import aggregate_outcomes, score_signal
from .bars import Timeframe, fetch_intraday
from .live import LiveEvaluation, SignalState, evaluate_live
from .reversal import ScenarioConfig, default_ensemble, detect_session


@dataclass(frozen=True)
class RankedSignal:
    """A live evaluation enriched with its scenario's historical edge."""

    live: LiveEvaluation
    hist_origin_retest: float  # historical hit-rate point estimate
    hist_retest_low: float  # Wilson lower bound
    hist_retest_high: float
    hist_n: int  # historical sample size
    hist_sufficient: bool  # enough history to trust the edge
    edge_score: float  # composite ranking score
    agreement: int = 1  # how many scenario configs fired this same
    # structure (origin/counter/entry). Higher
    # agreement = more robust, less parameter-luck.
    agreeing_scenarios: tuple[str, ...] = ()

    @property
    def reward_risk(self) -> float:
        """Origin distance / stop distance from current price."""
        lv = self.live
        risk = abs(lv.last_price - lv.counter_extreme)
        reward = abs(lv.origin_extreme - lv.last_price)
        return reward / risk if risk > 0 else float("nan")


def _build_history(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe,
    configs: list[ScenarioConfig],
    history_days: int,
) -> dict[tuple[str, str], object]:
    """Backtest the scenarios over recent history; return scenario scoreboard."""
    to_date = pd.Timestamp.now().normalize()
    from_date = to_date - pd.Timedelta(days=history_days)
    bars = fetch_intraday(
        client,
        symbol,
        timeframe,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
    )
    if bars.empty:
        return {}

    from .reversal import _prepare_session

    outcomes = []
    for _, session_bars in bars.groupby("date"):
        if len(session_bars) < 5:
            continue
        prepped = _prepare_session(session_bars)
        for sig in detect_session(symbol, session_bars, configs):
            outcomes.append(score_signal(sig, prepped))
    return aggregate_outcomes(outcomes)


def _edge_score(
    state: SignalState,
    hist_point: float,
    hist_low: float,
    sufficient: bool,
    reward_risk: float,
) -> float:
    """
    Composite ranking. Rewards (a) actionability (confirmed > forming > watch),
    (b) a high AND statistically trustworthy historical retest rate -- we use the
    Wilson LOWER bound, not the point estimate, so thin-sample optimism is
    penalised, (c) favourable reward:risk. Insufficient history is discounted,
    not zeroed, so new setups still surface but rank below proven ones.
    """
    state_w = {"confirmed": 1.0, "forming": 0.6, "watch": 0.35, "none": 0.0}[state.value]
    edge = hist_low if sufficient else hist_point * 0.5
    rr = min(reward_risk, 5.0) / 5.0 if reward_risk == reward_risk else 0.0
    return round(state_w * (0.6 * edge + 0.4 * rr), 4)


def scan_symbol(
    client: FMPClient,
    symbol: str,
    timeframe: Timeframe,
    configs: list[ScenarioConfig] | None = None,
    history_days: int = 60,
    states: tuple[SignalState, ...] = (
        SignalState.CONFIRMED,
        SignalState.FORMING,
        SignalState.WATCH,
    ),
) -> list[RankedSignal]:
    """
    Scan one symbol: backtest -> live-evaluate -> rank.

    Returns every live signal in the requested states, each tagged with its
    scenario's historical edge, sorted by edge_score descending.
    """
    symbol = symbol.strip().upper()
    configs = configs or default_ensemble()
    scoreboard = _build_history(client, symbol, timeframe, configs, history_days)

    # Today's session (the live one): fetch a short recent window and take the
    # last session present.
    to_date = pd.Timestamp.now().normalize()
    from_date = to_date - pd.Timedelta(days=5)
    recent = fetch_intraday(
        client,
        symbol,
        timeframe,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=(to_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if recent.empty:
        return []
    last_session_date = recent["date"].max()
    session_bars = recent[recent["date"] == last_session_date]

    ranked: list[RankedSignal] = []
    for cfg in configs:
        evals = evaluate_live(
            symbol,
            session_bars,
            cfg,
            now=pd.Timestamp.now(),
            bar_minutes=timeframe.minutes,
        )
        for ev in evals:
            if ev.state not in states:
                continue
            score = scoreboard.get((cfg.name, ev.side.value))
            if score is not None:
                est = score.target_hit_rates["origin_retest"]
                hist_point, hist_low, hist_high = est.point, est.wilson_low, est.wilson_high
                hist_n, suff = est.trials, est.sufficient
            else:
                hist_point = hist_low = hist_high = float("nan")
                hist_n, suff = 0, False

            rr = (
                abs(ev.origin_extreme - ev.last_price) / abs(ev.last_price - ev.counter_extreme)
                if abs(ev.last_price - ev.counter_extreme) > 0
                else float("nan")
            )
            edge = _edge_score(
                ev.state,
                hist_point if hist_point == hist_point else 0.0,
                hist_low if hist_low == hist_low else 0.0,
                suff,
                rr,
            )
            ranked.append(
                RankedSignal(
                    live=ev,
                    hist_origin_retest=hist_point,
                    hist_retest_low=hist_low,
                    hist_retest_high=hist_high,
                    hist_n=hist_n,
                    hist_sufficient=suff,
                    edge_score=edge,
                )
            )

    ranked.sort(key=lambda r: r.edge_score, reverse=True)
    return _deduplicate(ranked)


def _deduplicate(signals: list[RankedSignal]) -> list[RankedSignal]:
    """
    Collapse signals that describe the SAME live structure into one, counting
    scenario agreement. Many ensemble configs fire on identical origin/counter/
    entry levels; showing them as separate rows is noise. We group by (side,
    state, rounded origin, rounded counter, rounded entry) and keep the
    highest-edge representative, annotated with how many scenarios agreed and a
    small bonus for that agreement (robustness across parameterisations).
    """
    if not signals:
        return signals

    groups: dict[tuple, list[RankedSignal]] = {}
    for r in signals:
        lv = r.live
        key = (
            lv.side.value,
            lv.state.value,
            round(lv.origin_extreme, 2),
            round(lv.counter_extreme, 2),
            round(lv.last_price, 2),
        )
        groups.setdefault(key, []).append(r)

    deduped: list[RankedSignal] = []
    for grp in groups.values():
        grp.sort(key=lambda r: r.edge_score, reverse=True)
        best = grp[0]
        scenarios = tuple(sorted({r.live.scenario for r in grp}))
        n_agree = len(scenarios)
        # Agreement bonus: up to +15% for broad consensus, capped.
        bonus = min(0.15, 0.02 * (n_agree - 1))
        new_score = round(best.edge_score * (1.0 + bonus), 4)
        deduped.append(
            RankedSignal(
                live=best.live,
                hist_origin_retest=best.hist_origin_retest,
                hist_retest_low=best.hist_retest_low,
                hist_retest_high=best.hist_retest_high,
                hist_n=best.hist_n,
                hist_sufficient=best.hist_sufficient,
                edge_score=new_score,
                agreement=n_agree,
                agreeing_scenarios=scenarios,
            )
        )
    deduped.sort(key=lambda r: r.edge_score, reverse=True)
    return deduped


def scan_watchlist(
    client: FMPClient,
    symbols: list[str],
    timeframe: Timeframe,
    configs: list[ScenarioConfig] | None = None,
    history_days: int = 60,
    states: tuple[SignalState, ...] = (
        SignalState.CONFIRMED,
        SignalState.FORMING,
        SignalState.WATCH,
    ),
) -> dict[str, list[RankedSignal]]:
    """Scan many symbols; return {symbol: ranked signals}. Symbols with errors
    are skipped with the error recorded under an empty list (caller reports)."""
    out: dict[str, list[RankedSignal]] = {}
    for sym in symbols:
        try:
            out[sym] = scan_symbol(client, sym, timeframe, configs, history_days, states)
        except Exception as exc:  # one bad ticker must not kill the scan
            import logging

            logging.getLogger("engine.intraday.scanner").warning("Scan failed for %s: %s", sym, exc)
            out[sym] = []
    return out
