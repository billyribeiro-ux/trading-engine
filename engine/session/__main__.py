"""
Session dissection CLI.

Usage
-----
    export FMP_API_KEY=your_rotated_key
    python -m engine.session TSLA                      # most recent session, 1min
    python -m engine.session TSLA --date 2026-06-18    # a specific session
    python -m engine.session TSLA --timeframe 5min     # Premium-tier resolution
    python -m engine.session TSLA --date 2026-06-18 --verbose

Prints the full event narration for one real trading session: the flush, the
reversal, VWAP reclaim/loss, retest holds/fails, HOD/LOD tests, and the leg-by-
leg structure -- the chart read back as named, measured events. Run it on a
session you can see on your own chart and compare.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ..data.client import AuthError, FMPClient, FMPError
from ..intraday.bars import Timeframe, selectable_timeframes
from .report import render_report
from .runner import dissect_real_session


def _print_dissection(d, verbose: bool) -> None:
    line = "=" * 88
    print(line)
    print(d.narrate())
    print(line)

    if verbose:
        print("\nVWAP EVENTS (detail):")
        for e in d.vwap_events:
            print(
                f"  {e.time.strftime('%H:%M')} {e.type.value:<12} "
                f"px={e.price:.2f} vwap={e.vwap:.2f} "
                f"-> {e.outcome_atr:+.2f} ATR over {e.horizon_bars} bars"
            )
        print("\nLEVEL EVENTS (detail):")
        for e in d.level_events:
            verdict = "HELD" if e.held else "BROKE"
            print(
                f"  {e.time.strftime('%H:%M')} {e.kind.value:<16} @ {e.level:.2f} "
                f"{verdict} -> {e.outcome_atr:+.2f} ATR"
            )

    print("\nSUMMARY:")
    for k, v in d.summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.session",
        description="Dissect one real trading session into named, measured events.",
    )
    parser.add_argument("symbol", help="Ticker, e.g. TSLA")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest).")
    parser.add_argument(
        "--timeframe",
        default="1min",
        choices=[t.value for t in Timeframe],
        help="Bar resolution (default 1min; 5min works on Premium).",
    )
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument("--tier", choices=["free", "starter", "premium", "ultimate"], default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug-scales",
        action="store_true",
        help="Dump raw + merged legs at every scale (for structural diagnosis).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("ERROR: set FMP_API_KEY in your environment.", file=sys.stderr)
        return 2

    tf = Timeframe(args.timeframe)
    try:
        client = FMPClient(api_key, tier=args.tier)
        avail = selectable_timeframes(client.tier)[tf.value]
        if not avail["available"]:
            print(
                f"ERROR: {tf.value} needs {avail['min_tier']} tier "
                f"({avail['how']}); your tier is {client.tier.name}. "
                f"Try --timeframe 5min.",
                file=sys.stderr,
            )
            return 4
        if args.verbose:
            print(f"Tier {client.tier.name} | {tf.value} ({avail['how']})")
        session, d, nested = dissect_real_session(
            client,
            args.symbol,
            timeframe=tf,
            on_date=args.date,
            history_days=args.history_days,
        )
    except AuthError as exc:
        print(f"AUTH ERROR: {exc}", file=sys.stderr)
        return 3
    except (FMPError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(render_report(session, d, nested))

    if args.debug_scales:
        from .pivots import build_skeleton, decompose, merge_insignificant_swings

        dec = decompose(session)
        print("\n" + "=" * 78)
        print("  DEBUG: hierarchical skeleton (major split at structural pullbacks)")
        print("=" * 78)
        skel = build_skeleton(dec, major_scale=d.scale_atr)
        for lg in skel:
            dur = int((lg.end_time - lg.start_time).total_seconds() / 60)
            print(
                f"     {lg.direction:>4} {lg.start_time.strftime('%H:%M')}"
                f"->{lg.end_time.strftime('%H:%M')} ({dur:>3}m,{lg.bars:>2}b) "
                f"{lg.start_price:7.2f}->{lg.end_price:7.2f} ({lg.magnitude:5.2f}pt)"
            )
        print("\n" + "=" * 78)
        print("  DEBUG: raw vs merged legs at each scale")
        print("=" * 78)
        for sc in dec.scales:
            raw = dec.legs_by_scale.get(sc, [])
            merged = merge_insignificant_swings(raw)
            print(
                f"\n  scale {sc:.2f} ATR:  raw={len(raw)}  merged={len(merged)}"
                + ("   <-- PRIMARY" if sc == d.scale_atr else "")
            )
            for lg in merged:
                dur = int((lg.end_time - lg.start_time).total_seconds() / 60)
                print(
                    f"     {lg.direction:>4} {lg.start_time.strftime('%H:%M')}"
                    f"->{lg.end_time.strftime('%H:%M')} ({dur:>3}m,{lg.bars:>2}b) "
                    f"{lg.start_price:7.2f}->{lg.end_price:7.2f} "
                    f"({lg.magnitude:5.2f}pt)"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
