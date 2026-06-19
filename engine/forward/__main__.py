"""
Live forward-test CLI — accumulate real evidence for a validated edge.

    python -m engine.forward scan    --scanner swing --symbols AAPL,MSFT,NVDA
    python -m engine.forward resolve --scanner swing
    python -m engine.forward report

Schedule `scan` daily (after the close) and `resolve` daily on cron/launchd to
turn the historically-validated edge into live realized-vs-validated numbers.
Signals are emitted only if the pooled config STILL passes the forward gate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_DEFAULT_UNIVERSE = "AAPL,MSFT,NVDA,AMZN,META,GOOGL,KO,PEP,DIS,CSCO,INTC,ORCL,CRM,ADBE,COST,HD"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="engine.forward")
    p.add_argument("action", choices=["scan", "resolve", "report", "gauntlet"])
    p.add_argument("--symbols", default=_DEFAULT_UNIVERSE)
    p.add_argument("--scanner", default="swing", choices=["intraday", "swing", "portfolio"])
    p.add_argument("--model", default="gbt", choices=["logistic", "gbt"])
    p.add_argument("--direction", default="long", choices=["long", "short"])
    p.add_argument("--proba", type=float, default=0.55)
    p.add_argument("--lookback", type=int, default=2920)
    p.add_argument("--journal", default=None)
    args = p.parse_args(argv)

    from .journal import SignalJournal
    from .live import _DEFAULT_JOURNAL, LiveConfig, _config_for, resolve, scan_and_log

    jpath = Path(args.journal).expanduser() if args.journal else _DEFAULT_JOURNAL
    journal = SignalJournal(jpath)

    if args.action == "report":
        s = journal.summary()
        print(
            f"journal {jpath}\n"
            f"  open={s.n_open}  resolved={s.n_resolved}\n"
            f"  realized_R (net) = {s.realized_mean_r:+.3f}   hit_rate = {s.realized_hit_rate:.2f}\n"
            f"  carried validated edge_R = {s.mean_validated_edge_r:+.3f}\n"
            f"  exits = {s.by_reason}"
        )
        return 0

    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set")
        return 2
    from ..data.client import FMPClient

    client = FMPClient(key)
    live = LiveConfig(
        symbols=tuple(x.strip().upper() for x in args.symbols.split(",") if x.strip()),
        scanner=args.scanner,
        model_kind=args.model,
        direction=args.direction,
        proba_threshold=args.proba,
        lookback_days=args.lookback,
        journal_path=jpath,
    )

    if args.action == "scan":
        sigs = scan_and_log(client, live, journal)
        print(f"logged {len(sigs)} {live.direction} {live.scanner} signals to {jpath}")
        for s in sigs[:25]:
            print(
                f"  {s.symbol:<6} {s.event_type:<14} entry={s.entry:.2f} stop={s.stop:.2f} "
                f"target={s.target:.2f} R:R={s.rr:.2f} p={s.probability:.2f}"
            )
    elif args.action == "gauntlet":
        from .gauntlet import render_gauntlet, run_gauntlet

        config = _config_for(client, live)
        v = run_gauntlet(live.symbols, config, models=("logistic", "gbt"), direction=live.direction)
        print(render_gauntlet(v))
    else:  # resolve
        rows = resolve(client, live, journal)
        nres = sum(1 for r in rows if r.get("status") == "resolved")
        print(f"resolved journal: {nres} resolved / {len(rows)} total at {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
