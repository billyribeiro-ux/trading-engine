"""
Regime-conditioned edge estimation.

Joins each signal's outcome to its session regime, then measures the edge per
(scenario, side, regime). The result is what the live scanner consults: given
today's regime, what is THIS scenario's measured net expectancy and retest rate
in that regime -- not a cross-regime blend that is wrong in every regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..gaps.statistics import bootstrap_mean_ci, estimate_proportion
from .backtest import SignalOutcome
from .regime import SessionRegime


@dataclass(frozen=True)
class RegimeEdge:
    scenario: str
    side: str
    regime_label: str  # e.g. "range/normal_vol"
    n: int
    net_expectancy_r: float
    net_ci: tuple[float, float]
    origin_retest_rate: object  # ProportionEstimate
    win_rate: object  # ProportionEstimate
    sufficient: bool


def condition_on_regime(
    outcomes: list[SignalOutcome],
    regimes: dict[pd.Timestamp, SessionRegime],
    by_volatility: bool = True,
) -> dict[tuple[str, str, str], RegimeEdge]:
    """
    Compute per-regime edges.

    Parameters
    ----------
    outcomes : cost-aware SignalOutcomes.
    regimes  : session -> SessionRegime (from classify_all_sessions).
    by_volatility : if True, regime label combines directional + volatility
        (finer buckets, needs more data). If False, directional only (coarser,
        more samples per bucket). The scanner can request either granularity.

    Returns {(scenario, side, regime_label): RegimeEdge}.
    """
    if not outcomes:
        return {}

    rows = []
    for o in outcomes:
        reg = regimes.get(pd.Timestamp(o.signal.session))
        if reg is None:
            continue
        label = reg.label if by_volatility else reg.directional.value
        rows.append(
            {
                "scenario": o.signal.scenario,
                "side": o.signal.side.value,
                "regime": label,
                "net_r": o.net_r,
                "retest": o.targets_hit.get("origin_retest", False),
            }
        )
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    # Global priors for shrinkage.
    global_retest = df["retest"].mean()
    global_win = (df["net_r"] > 0).mean()

    out: dict[tuple[str, str, str], RegimeEdge] = {}
    for (scenario, side, regime), sub in df.groupby(["scenario", "side", "regime"]):
        n = len(sub)
        net = sub["net_r"].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_ci(net)
        retest = estimate_proportion(
            int(sub["retest"].sum()),
            n,
            prior_mean=float(global_retest),
            prior_strength=12,
        )
        win = estimate_proportion(
            int((sub["net_r"] > 0).sum()),
            n,
            prior_mean=float(global_win),
            prior_strength=12,
        )
        out[(scenario, side, regime)] = RegimeEdge(
            scenario=scenario,
            side=side,
            regime_label=regime,
            n=n,
            net_expectancy_r=float(mean),
            net_ci=(float(lo), float(hi)),
            origin_retest_rate=retest,
            win_rate=win,
            sufficient=retest.sufficient,
        )
    return out
