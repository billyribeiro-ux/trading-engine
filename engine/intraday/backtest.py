"""
Reversal outcome backtester with a target ladder.

When a reversal fires, "did it work?" has no single answer -- it has a ladder of
targets reached in sequence. This scores each signal against all of them on the
bars *after* the signal, within the same session:

    VWAP            -- the mean-reversion magnet
    ORIGIN_RETEST   -- the session high/low that the structure aims to retest
                       (your core thesis: sell off, reverse, test the same HOD)
    HALF_RETEST     -- 50% of the way from entry to the origin extreme
    OPPOSITE_EXT    -- the counter-extreme reached before the reversal
    ATR_1R/2R/3R    -- fixed R-multiples from entry

The stop is the counter-extreme (the low that held on a bullish reversal, the
high that capped on a bearish one). For each signal we record which targets were
hit, the order, max favorable / adverse excursion, and bars-to-target. These
feed the same Wilson/Bayes statistics as the gap module, scored per scenario.

The ladder is a list -- adding a target (POC/VAH/VAL, PDH/PDL) is one entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .costs import CostModel, Side
from .reversal import ReversalSide, ReversalSignal


class Target(str, Enum):
    VWAP = "vwap"
    ORIGIN_RETEST = "origin_retest"
    HALF_RETEST = "half_retest"
    OPPOSITE_EXT = "opposite_ext"
    ATR_1R = "atr_1r"
    ATR_2R = "atr_2r"
    ATR_3R = "atr_3r"


@dataclass(frozen=True)
class TargetLevel:
    target: Target
    price: float


@dataclass(frozen=True)
class SignalOutcome:
    """Scored result for one reversal signal."""

    signal: ReversalSignal
    stopped_out: bool
    stop_price: float
    bars_held: int
    mfe: float  # max favorable excursion, price units
    mae: float  # max adverse excursion, price units
    mfe_r: float  # MFE in R (risk = entry->stop distance)
    mae_r: float
    targets_hit: dict[str, bool]  # target -> reached before stop/EOD
    bars_to_target: dict[str, int]  # target -> bar count to first touch
    final_return: float  # entry -> session close, signed to side
    # --- net-of-cost trade simulation (defined exit policy) ---
    exit_reason: str  # 'target' | 'stop' | 'session_close'
    exit_price: float  # the price level the exit policy used
    gross_r: float  # gross R from entry to exit (no costs)
    net_r: float  # net R after spread+slippage+commission
    net_pnl_per_share: float  # $ per share, net
    cost_r: float  # R lost to friction (gross_r - net_r)


def _build_ladder(sig: ReversalSignal) -> list[TargetLevel]:
    """Construct the target ladder for a signal, in the reversal direction."""
    entry = sig.entry_price
    origin = sig.origin_extreme
    atr = sig.atr_at_signal
    half = entry + (origin - entry) * 0.5

    if sig.side is ReversalSide.BULLISH:
        r1, r2, r3 = entry + atr, entry + 2 * atr, entry + 3 * atr
    else:
        r1, r2, r3 = entry - atr, entry - 2 * atr, entry - 3 * atr

    return [
        TargetLevel(Target.VWAP, sig.vwap_at_signal),
        TargetLevel(Target.ORIGIN_RETEST, origin),
        TargetLevel(Target.HALF_RETEST, half),
        TargetLevel(Target.OPPOSITE_EXT, sig.counter_extreme),
        TargetLevel(Target.ATR_1R, r1),
        TargetLevel(Target.ATR_2R, r2),
        TargetLevel(Target.ATR_3R, r3),
    ]


def score_signal(
    sig: ReversalSignal,
    session_bars: pd.DataFrame,
    cost_model: CostModel | None = None,
    exit_target: Target = Target.ORIGIN_RETEST,
    shares: float = 100.0,
) -> SignalOutcome:
    """
    Score one signal over the bars after it within its session, with a defined
    exit policy and realistic costs.

    Exit policy: the trade exits at `exit_target` if that level is reached before
    the stop; at the stop (counter-extreme) if hit first; otherwise at the
    session close (time stop). Intrabar ambiguity is resolved pessimistically: if
    a bar's range spans both stop and target, the STOP is taken first. Costs
    (spread, slippage, commission) are applied at the actual fill via cost_model;
    a touched stop fills WORSE than its trigger (it becomes a market order).

    Reports both gross_r and net_r so the friction is explicit -- the gap between
    them is exactly what a live edge must clear to be real.
    """
    cost_model = cost_model or CostModel()
    df = session_bars.reset_index(drop=True)
    start = sig.signal_index + 1
    forward = df.iloc[start:]
    side = sig.side
    entry = sig.entry_price
    stop = sig.counter_extreme
    atr = sig.atr_at_signal
    risk = abs(entry - stop)
    if risk <= 0:
        risk = atr  # degenerate guard

    ladder = _build_ladder(sig)
    target_price = next((t.price for t in ladder if t.target is exit_target), sig.origin_extreme)
    targets_hit = {t.target.value: False for t in ladder}
    bars_to_target = {t.target.value: -1 for t in ladder}

    mfe = 0.0
    mae = 0.0
    stopped = False
    bars_held = 0
    exit_reason = "session_close"
    exit_price = float(forward["close"].iloc[-1]) if not forward.empty else entry

    final_close = float(forward["close"].iloc[-1]) if not forward.empty else entry

    for k, (_, bar) in enumerate(forward.iterrows(), start=1):
        hi, lo = float(bar["high"]), float(bar["low"])
        bars_held = k

        if side is ReversalSide.BULLISH:
            fav = hi - entry
            adv = entry - lo
            stop_hit = lo <= stop
            target_hit = hi >= target_price
        else:
            fav = entry - lo
            adv = hi - entry
            stop_hit = hi >= stop
            target_hit = lo <= target_price
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Record all ladder touches (for hit-rate stats), still pessimistic on stop.
        if not stop_hit:
            for tl in ladder:
                key = tl.target.value
                if targets_hit[key]:
                    continue
                reached = (hi >= tl.price) if side is ReversalSide.BULLISH else (lo <= tl.price)
                if reached:
                    targets_hit[key] = True
                    bars_to_target[key] = k

        # Exit decision (pessimistic: stop before target on a spanning bar).
        if stop_hit:
            stopped = True
            exit_reason = "stop"
            exit_price = stop
            break
        if target_hit:
            exit_reason = "target"
            exit_price = target_price
            break

    # Cost-aware P&L.
    cm_side = Side.LONG if side is ReversalSide.BULLISH else Side.SHORT
    pnl = cost_model.round_trip_pnl(
        side=cm_side,
        entry_signal=entry,
        exit_price=exit_price,
        atr=atr,
        shares=shares,
        exit_is_stop=(exit_reason == "stop"),
    )
    gross_r = (
        (exit_price - entry) if side is ReversalSide.BULLISH else (entry - exit_price)
    ) / risk
    net_r = pnl["net_pnl_per_share"] / risk
    final_return = (
        (final_close - entry) / entry
        if side is ReversalSide.BULLISH
        else (entry - final_close) / entry
    )

    return SignalOutcome(
        signal=sig,
        stopped_out=stopped,
        stop_price=stop,
        bars_held=bars_held,
        mfe=mfe,
        mae=mae,
        mfe_r=mfe / risk,
        mae_r=mae / risk,
        targets_hit=targets_hit,
        bars_to_target=bars_to_target,
        final_return=final_return,
        exit_reason=exit_reason,
        exit_price=exit_price,
        gross_r=gross_r,
        net_r=net_r,
        net_pnl_per_share=pnl["net_pnl_per_share"],
        cost_r=gross_r - net_r,
    )


@dataclass(frozen=True)
class ScenarioScore:
    """Aggregated outcomes for one scenario across all its signals."""

    scenario: str
    side: str
    n: int
    target_hit_rates: dict[str, object]  # target -> ProportionEstimate
    stop_rate: object  # ProportionEstimate
    median_bars_to_origin: float
    mean_mfe_r: float
    mean_mae_r: float
    expectancy_r: float  # mean final_return / mean risk proxy
    # --- cost-aware, defined-exit expectancy ---
    gross_expectancy_r: float  # mean gross R at the defined exit
    net_expectancy_r: float  # mean net R after all friction
    net_expectancy_ci: tuple[float, float]  # bootstrap 95% CI on net R
    cost_drag_r: float  # mean R lost to friction
    win_rate_net: object  # ProportionEstimate of net_r > 0


def aggregate_outcomes(
    outcomes: list[SignalOutcome],
) -> dict[tuple[str, str], ScenarioScore]:
    """
    Group outcomes by (scenario, side) and compute honest hit-rates per target.

    Uses the gap module's proportion estimator: Wilson interval + Bayesian
    shrinkage + sufficiency flag. The base rate prior for each target is the
    pooled hit-rate across all scenarios for that target, so a thin scenario is
    shrunk toward the global behaviour rather than toward 50% or toward noise.
    """
    from ..gaps.statistics import bootstrap_mean_ci, estimate_proportion

    if not outcomes:
        return {}

    rows = []
    for o in outcomes:
        rows.append(
            {
                "scenario": o.signal.scenario,
                "side": o.signal.side.value,
                "stopped": o.stopped_out,
                "mfe_r": o.mfe_r,
                "mae_r": o.mae_r,
                "final_return": o.final_return,
                "gross_r": o.gross_r,
                "net_r": o.net_r,
                "cost_r": o.cost_r,
                "bars_to_origin": o.bars_to_target.get("origin_retest", -1),
                **{f"hit_{k}": v for k, v in o.targets_hit.items()},
            }
        )
    df = pd.DataFrame(rows)

    target_keys = [c[4:] for c in df.columns if c.startswith("hit_")]
    # Global priors per target.
    global_prior = {t: df[f"hit_{t}"].mean() for t in target_keys}
    global_stop_prior = df["stopped"].mean()
    global_winrate = (df["net_r"] > 0).mean()

    out: dict[tuple[str, str], ScenarioScore] = {}
    for (scenario, side), sub in df.groupby(["scenario", "side"]):
        n = len(sub)
        hit_rates = {}
        for t in target_keys:
            hits = int(sub[f"hit_{t}"].sum())
            hit_rates[t] = estimate_proportion(
                hits, n, prior_mean=float(global_prior[t]), prior_strength=15
            )
        stop_est = estimate_proportion(
            int(sub["stopped"].sum()),
            n,
            prior_mean=float(global_stop_prior),
            prior_strength=15,
        )
        bto = sub.loc[sub["bars_to_origin"] > 0, "bars_to_origin"]
        median_bto = float(bto.median()) if not bto.empty else float("nan")
        exp_mean, _, _ = bootstrap_mean_ci(sub["final_return"].to_numpy())
        net_mean, net_lo, net_hi = bootstrap_mean_ci(sub["net_r"].to_numpy())
        gross_mean, _, _ = bootstrap_mean_ci(sub["gross_r"].to_numpy())
        win_net = estimate_proportion(
            int((sub["net_r"] > 0).sum()),
            n,
            prior_mean=float(global_winrate),
            prior_strength=15,
        )

        out[(scenario, side)] = ScenarioScore(
            scenario=scenario,
            side=side,
            n=n,
            target_hit_rates=hit_rates,
            stop_rate=stop_est,
            median_bars_to_origin=median_bto,
            mean_mfe_r=float(sub["mfe_r"].mean()),
            mean_mae_r=float(sub["mae_r"].mean()),
            expectancy_r=float(exp_mean),
            gross_expectancy_r=float(gross_mean),
            net_expectancy_r=float(net_mean),
            net_expectancy_ci=(float(net_lo), float(net_hi)),
            cost_drag_r=float(sub["cost_r"].mean()),
            win_rate_net=win_net,
        )
    return out
