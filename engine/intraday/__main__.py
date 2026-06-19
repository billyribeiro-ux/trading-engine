"""
Live intraday reversal scanner CLI.

Usage
-----
    export FMP_API_KEY=your_key
    python -m engine.intraday AAPL --timeframe 5min
    python -m engine.intraday AAPL MSFT NVDA --timeframe 5min --states confirmed,forming
    python -m engine.intraday TSLA --timeframe 15min --history-days 90 --verbose

Runs the historical backtest per scenario, then scans the live session and lists
every signal (confirmed/forming/watch) ranked by measured edge. The historical
origin-retest hit-rate and its confidence interval are shown next to each live
signal so you act on proven edge, not a bare alert.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..data.client import AuthError, FMPClient, FMPError
from .bars import Timeframe, selectable_timeframes
from .live import SignalState
from .scanner import RankedSignal, scan_watchlist


def _fmt_signal(r: RankedSignal) -> str:
    lv = r.live
    side = "BULL" if lv.side.value == "bullish" else "BEAR"
    if r.hist_sufficient:
        edge = (
            f"hist retest {r.hist_origin_retest:.0%} "
            f"[{r.hist_retest_low:.0%}-{r.hist_retest_high:.0%}] n={r.hist_n}"
        )
    elif r.hist_n > 0:
        edge = f"hist retest {r.hist_origin_retest:.0%} n={r.hist_n} [thin]"
    else:
        edge = "no history"
    agree = f" x{r.agreement}" if r.agreement > 1 else ""
    return (
        f"  [{r.edge_score:5.3f}] {lv.state.value.upper():<9} {side}{agree} "
        f"{lv.scenario:<34} "
        f"px={lv.last_price:.2f} origin={lv.origin_extreme:.2f} "
        f"stop={lv.counter_extreme:.2f} R:R={r.reward_risk:.1f} | {edge}\n"
        f"            {lv.note}" + (f"  ({r.agreement} scenarios agree)" if r.agreement > 1 else "")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.intraday",
        description="Live intraday reversal scanner ranked by measured edge.",
    )
    parser.add_argument("symbols", nargs="+", help="One or more tickers.")
    parser.add_argument(
        "--timeframe",
        default="5min",
        choices=[t.value for t in Timeframe],
        help="Signal timeframe (default 5min). Non-native ones resample/gate.",
    )
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument(
        "--states",
        default="confirmed,forming,watch",
        help="Comma list of states to show: confirmed,forming,watch.",
    )
    parser.add_argument("--tier", choices=["free", "starter", "premium", "ultimate"], default=None)
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
            "ERROR: no FMP API key — set FMP_API_KEY or save one in the dashboard Settings.",
            file=sys.stderr,
        )
        return 2

    tf = Timeframe(args.timeframe)
    want_states = tuple(SignalState(s.strip()) for s in args.states.split(",") if s.strip())

    try:
        client = FMPClient(api_key, tier=args.tier)
        tf_avail = selectable_timeframes(client.tier)[tf.value]
        if not tf_avail["available"]:
            print(
                f"ERROR: {tf.value} requires {tf_avail['min_tier']} tier "
                f"({tf_avail['how']}). Your tier: {client.tier.name}.\n"
                f"Upgrade, or choose a native timeframe.",
                file=sys.stderr,
            )
            return 4
        if args.verbose:
            print(f"Tier {client.tier.name} | timeframe {tf.value} ({tf_avail['how']})")

        results = scan_watchlist(
            client,
            args.symbols,
            tf,
            history_days=args.history_days,
            states=want_states,
        )
    except AuthError as exc:
        print(f"AUTH ERROR: {exc}", file=sys.stderr)
        return 3
    except (FMPError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    line = "=" * 104
    print(line)
    print(f"LIVE REVERSAL SCAN | {tf.value} | states: {args.states}")
    print(line)

    any_hit = False

    # Sort symbols by their best signal's edge.
    def best(sym: str) -> float:
        sigs = results.get(sym, [])
        return sigs[0].edge_score if sigs else -1.0

    for sym in sorted(args.symbols, key=best, reverse=True):
        sigs = results.get(sym, [])
        if not sigs:
            if args.verbose:
                print(f"\n{sym}: no qualifying signals")
            continue
        any_hit = True
        print(f"\n{sym}  ({len(sigs)} signal(s))")
        for r in sigs:
            print(_fmt_signal(r))

    if not any_hit:
        print("\nNo confirmed/forming/watch reversals in the current session.")
    print(line)
    print(
        "edge_score ranks by actionability x historical retest edge (Wilson lower "
        "bound) x reward:risk. [thin] = insufficient history; treat as unproven."
    )
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
