"""
Fundamental features with AS-OF-date causality (Scanner #3, the deep layer).

THE FUNDAMENTAL CAUSALITY TRAP: an income statement's `date` is the fiscal period
END (e.g. 2026-03-28), but the figure is not PUBLIC until its `acceptedDate`
(2026-05-01 here — five weeks later). Using it before acceptedDate is lookahead,
the worst kind. Earnings actuals are known on their announcement `date`. This
module joins every fundamental AS OF the date it became KNOWN, never the period it
describes — so a feature at a decision bar uses only reports filed on or before
that bar. A report filed AFTER the bar can never influence it.

Returns a sparse dict (only the features knowable as-of the date); missing ones
are mean-imputed downstream, so a name with no filings yet simply contributes no
fundamental signal rather than a leak or a fabricated zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.client import FMPClient


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class _Income:
    asof: pd.Timestamp  # acceptedDate -> when the filing became public
    eps: float
    revenue: float


@dataclass(frozen=True)
class _Earnings:
    asof: pd.Timestamp  # announcement date
    eps_actual: float
    eps_est: float
    rev_actual: float
    rev_est: float


@dataclass
class FundamentalSeries:
    """As-of-joinable fundamentals for one symbol (sorted ascending by as-of)."""

    symbol: str
    income: list[_Income] = field(default_factory=list)
    earnings: list[_Earnings] = field(default_factory=list)

    @classmethod
    def from_client(cls, client: FMPClient, symbol: str, limit: int = 40) -> FundamentalSeries:
        income: list[_Income] = []
        try:
            df = client.fetch(
                "income_statement", symbol=symbol, params={"period": "quarter", "limit": limit}
            )
            for _, r in df.iterrows():
                # acceptedDate is the as-of date; fall back to filingDate, never `date`
                # (the fiscal period end) on its own.
                raw = r.get("acceptedDate") or r.get("filingDate")
                asof = pd.to_datetime(raw, errors="coerce")
                if pd.isna(asof):
                    continue
                income.append(
                    _Income(pd.Timestamp(asof).normalize(), _f(r.get("eps")), _f(r.get("revenue")))
                )
        except Exception:
            pass
        earnings: list[_Earnings] = []
        try:
            df = client.fetch("earnings_symbol", symbol=symbol, params={"limit": limit})
            for _, r in df.iterrows():
                asof = pd.to_datetime(r.get("date"), errors="coerce")
                if pd.isna(asof):
                    continue
                earnings.append(
                    _Earnings(
                        pd.Timestamp(asof).normalize(),
                        _f(r.get("epsActual")),
                        _f(r.get("epsEstimated")),
                        _f(r.get("revenueActual")),
                        _f(r.get("revenueEstimated")),
                    )
                )
        except Exception:
            pass
        income.sort(key=lambda x: x.asof)
        earnings.sort(key=lambda x: x.asof)
        return cls(symbol, income, earnings)

    def asof(self, date) -> dict[str, float]:
        """Fundamental features known on or before `date`. Reports filed after it
        are excluded — that exclusion IS the causality guarantee."""
        d = pd.Timestamp(date).normalize()
        out: dict[str, float] = {}

        known = [r for r in self.income if r.asof <= d]
        if known:
            latest = known[-1]
            out["fund_days_since_report"] = float((d - latest.asof).days)
            if len(known) >= 5:  # year-over-year = same quarter 4 reports earlier
                prior = known[-5]
                if np.isfinite(latest.eps) and np.isfinite(prior.eps) and abs(prior.eps) > 1e-9:
                    out["eps_yoy"] = (latest.eps - prior.eps) / abs(prior.eps)
                if np.isfinite(latest.revenue) and np.isfinite(prior.revenue) and prior.revenue > 0:
                    out["rev_yoy"] = (latest.revenue - prior.revenue) / prior.revenue

        kearn = [e for e in self.earnings if e.asof <= d]
        if kearn:
            e = kearn[-1]
            if np.isfinite(e.eps_actual) and np.isfinite(e.eps_est) and abs(e.eps_est) > 1e-9:
                out["eps_surprise"] = (e.eps_actual - e.eps_est) / abs(e.eps_est)
            if np.isfinite(e.rev_actual) and np.isfinite(e.rev_est) and e.rev_est > 0:
                out["rev_surprise"] = (e.rev_actual - e.rev_est) / e.rev_est
        return out

    def __bool__(self) -> bool:
        return bool(self.income or self.earnings)
