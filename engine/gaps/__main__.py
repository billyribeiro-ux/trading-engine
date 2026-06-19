"""
Gap analysis CLI.

Usage
-----
    export FMP_API_KEY=your_rotated_key
    python -m engine.gaps AAPL --lookback 10
    python -m engine.gaps TSLA --lookback 15 --min-gap-atr 0.25 --tier premium

The key is read ONLY from the environment, never passed on the command line
(argv is visible in process listings and shell history). Tier is auto-detected
unless pinned.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..data.client import AuthError, FMPClient, FMPError
from ..gaps.analysis import BucketStat, GapReport
from ..gaps.runner import run_gap_analysis


def _fmt_bucket(st: BucketStat) -> str:
    exp = st.expectancy_mean
    lo, hi = st.expectancy_ci
    return (
        f"  {st.label:<22} cont={st.continuation}  "
        f"fill={st.full_fill}  "
        f"E[ret]={exp:+.2%} [{lo:+.2%},{hi:+.2%}]"
    )


def _print_report(rep: GapReport) -> None:
    line = "=" * 100
    print(line)
    print(f"GAP ANALYSIS  |  {rep.symbol}  |  {rep.n_events} qualifying gaps")
    print(line)
    print(f"Base continuation: {rep.base_continuation}")
    print(f"Base full-fill:    {rep.base_full_fill}")

    def section(title: str, buckets: dict[str, BucketStat]) -> None:
        print(f"\n{title}")
        # Sort sufficient buckets first, then by n desc.
        for st in sorted(
            buckets.values(),
            key=lambda b: (not b.continuation.sufficient, -b.n),
        ):
            print(_fmt_bucket(st))

    section("BY DIRECTION", rep.by_direction)
    section("BY SIZE TIER (ATR)", rep.by_size)
    section("BY PRIOR TREND", rep.by_trend)
    section("BY WEEKDAY", rep.by_weekday)
    section("BY EARNINGS PROXIMITY", rep.by_earnings)

    print("\nSAME-SESSION FILL (Kaplan-Meier)")
    su, sd = rep.fill_survival_up, rep.fill_survival_down
    print(
        f"  up-gaps   : P(fill same session)={su.probability_by(1.0):.1%} "
        f"(events={su.n_events}, censored={su.n_censored})"
    )
    print(
        f"  down-gaps : P(fill same session)={sd.probability_by(1.0):.1%} "
        f"(events={sd.n_events}, censored={sd.n_censored})"
    )
    print(line)
    print(
        "Buckets flagged [INSUFFICIENT EVIDENCE] have too few events or too wide "
        "an interval to trade. They are shown, not hidden, so you can judge."
    )
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.gaps",
        description="Gap continuation/fill statistics for a ticker.",
    )
    parser.add_argument("symbol", help="Ticker, e.g. AAPL")
    parser.add_argument(
        "--lookback", type=float, default=10.0, help="Years of history (default 10)."
    )
    parser.add_argument(
        "--min-gap-atr",
        type=float,
        default=0.10,
        help="Ignore gaps smaller than this many ATR (default 0.10).",
    )
    parser.add_argument(
        "--tier",
        choices=["free", "starter", "premium", "ultimate"],
        default=None,
        help="Pin FMP tier; default auto-detect.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from ..settings import resolve_api_key

    api_key = resolve_api_key()
    if not api_key:
        print(
            "ERROR: no FMP API key. Set FMP_API_KEY or save one in the dashboard Settings.\n",
            file=sys.stderr,
        )
        return 2

    try:
        client = FMPClient(api_key, tier=args.tier)
        if args.verbose:
            caps = client.capabilities()
            print(f"Tier: {caps['tier']} ({caps['rate_limit_per_min']} req/min)")
            if caps["unavailable"]:
                print(f"Unavailable at this tier: {', '.join(caps['unavailable'])}")
        rep = run_gap_analysis(client, args.symbol, args.lookback, args.min_gap_atr)
    except AuthError as exc:
        print(f"AUTH ERROR: {exc}", file=sys.stderr)
        return 3
    except (FMPError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
