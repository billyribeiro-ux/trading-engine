"""
Statistical primitives used across the engine.

Every estimator here exists to stop small samples from lying. A raw hit-rate of
3/3 = 100% is meaningless; the same 75% from 400 trials is tradeable. These
functions make that distinction explicit in the output so no downstream report
can present a point estimate without its uncertainty.

No estimator here assumes normality of returns. Proportions use Wilson (exact-ish
for small n, unlike the Wald interval which gives nonsense near 0/1). Expectancy
uses the bootstrap (distribution-free). Time-to-event uses Kaplan-Meier (handles
censoring -- gaps that never filled within the observation window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

_Z_95: Final[float] = 1.959963984540054  # two-sided 95%


@dataclass(frozen=True)
class ProportionEstimate:
    """A probability with honest uncertainty and a verdict on sufficiency."""

    successes: int
    trials: int
    point: float  # raw successes/trials
    wilson_low: float  # Wilson score lower bound (95%)
    wilson_high: float  # Wilson score upper bound (95%)
    bayes_mean: float  # beta-binomial posterior mean (shrunk to prior)
    bayes_low: float  # posterior 2.5th percentile
    bayes_high: float  # posterior 97.5th percentile
    sufficient: bool  # trials >= min_trials AND interval width tolerable

    def __str__(self) -> str:
        flag = "" if self.sufficient else "  [INSUFFICIENT EVIDENCE]"
        return (
            f"{self.point:6.1%} (n={self.trials:>4}) "
            f"Wilson95[{self.wilson_low:.1%}, {self.wilson_high:.1%}] "
            f"Bayes[{self.bayes_low:.1%}, {self.bayes_high:.1%}]{flag}"
        )


def wilson_interval(successes: int, trials: int, z: float = _Z_95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    Correct near 0 and 1 and for small n, where the Wald interval fails. This is
    the workhorse for every hit-rate the engine reports.
    """
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    margin = z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def beta_binomial_posterior(
    successes: int,
    trials: int,
    prior_mean: float,
    prior_strength: float = 20.0,
) -> tuple[float, float, float]:
    """
    Posterior mean and 95% credible interval under a Beta prior.

    The prior is specified by its mean (e.g. the market-wide base rate for this
    outcome) and a 'strength' = pseudo-count. strength=20 means the prior carries
    the weight of 20 observations: a bucket with n=5 is dominated by the prior; a
    bucket with n=500 overwhelms it. This is the mechanism that turns a 3/3 cell
    into "still basically the base rate" instead of "100%".
    """
    prior_mean = min(max(prior_mean, 1e-6), 1 - 1e-6)
    alpha0 = prior_mean * prior_strength
    beta0 = (1 - prior_mean) * prior_strength
    alpha = alpha0 + successes
    beta = beta0 + (trials - successes)
    mean = alpha / (alpha + beta)
    low, high = stats.beta.ppf([0.025, 0.975], alpha, beta)
    return (float(mean), float(low), float(high))


def estimate_proportion(
    successes: int,
    trials: int,
    prior_mean: float,
    prior_strength: float = 20.0,
    min_trials: int = 30,
    max_interval_width: float = 0.35,
) -> ProportionEstimate:
    """
    Full proportion estimate with a sufficiency verdict.

    'sufficient' is True only when there are enough trials AND the Wilson
    interval is tight enough to act on. Both gates matter: 200 coin-flips give
    n>=30 but a wide interval; 30 lopsided trials give a tight interval but thin
    n. Reports key off this flag to avoid presenting noise as signal.
    """
    point = successes / trials if trials else float("nan")
    w_low, w_high = wilson_interval(successes, trials)
    b_mean, b_low, b_high = beta_binomial_posterior(successes, trials, prior_mean, prior_strength)
    sufficient = trials >= min_trials and (w_high - w_low) <= max_interval_width
    return ProportionEstimate(
        successes=successes,
        trials=trials,
        point=point,
        wilson_low=w_low,
        wilson_high=w_high,
        bayes_mean=b_mean,
        bayes_low=b_low,
        bayes_high=b_high,
        sufficient=sufficient,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """
    Distribution-free mean and percentile CI via the bootstrap.

    Used for expectancy (mean P&L per gap event), where the return distribution
    is fat-tailed and skewed -- a t-interval would understate tail risk.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo = (1 - ci) / 2
    hi = 1 - lo
    return (
        float(values.mean()),
        float(np.quantile(boot_means, lo)),
        float(np.quantile(boot_means, hi)),
    )


@dataclass(frozen=True)
class SurvivalCurve:
    """Kaplan-Meier estimate of time-to-event (e.g. time-to-fill)."""

    times: np.ndarray  # event times (sorted, unique)
    survival: np.ndarray  # P(not yet filled) at each time
    median_time: float  # first time survival <= 0.5, or nan if never
    n_events: int
    n_censored: int

    def probability_by(self, t: float) -> float:
        """P(event has occurred by time t) = 1 - S(t)."""
        if self.times.size == 0:
            return float("nan")
        idx = np.searchsorted(self.times, t, side="right") - 1
        if idx < 0:
            return 0.0
        return float(1.0 - self.survival[idx])


def kaplan_meier(
    durations: np.ndarray,
    observed: np.ndarray,
) -> SurvivalCurve:
    """
    Kaplan-Meier survival estimator.

    durations: time until event or censoring (e.g. minutes/bars until fill).
    observed:  1 if the event occurred, 0 if censored (gap never filled in the
               session -> we only know it lasted at least `duration`).

    Censoring is the whole point: a naive 'mean time to fill' silently drops the
    gaps that never filled, which are exactly the ones a trader most needs to
    know about. KM keeps them as right-censored observations.
    """
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=int)
    if durations.size == 0:
        return SurvivalCurve(np.array([]), np.array([]), float("nan"), 0, 0)

    order = np.argsort(durations)
    durations = durations[order]
    observed = observed[order]

    unique_times = np.unique(durations[observed == 1])
    if unique_times.size == 0:
        # No events at all -> survival stays at 1.
        return SurvivalCurve(
            np.array([durations.max()]),
            np.array([1.0]),
            float("nan"),
            0,
            int((observed == 0).sum()),
        )

    survival = np.empty(unique_times.size, dtype=float)
    s = 1.0
    for i, t in enumerate(unique_times):
        at_risk = (durations >= t).sum()
        events = ((durations == t) & (observed == 1)).sum()
        if at_risk > 0:
            s *= 1.0 - events / at_risk
        survival[i] = s

    below = np.where(survival <= 0.5)[0]
    median_time = float(unique_times[below[0]]) if below.size else float("nan")

    return SurvivalCurve(
        times=unique_times,
        survival=survival,
        median_time=median_time,
        n_events=int((observed == 1).sum()),
        n_censored=int((observed == 0).sum()),
    )
