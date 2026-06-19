"""
Serialize engine objects to plain JSON-ready dicts for the dashboard.

The CLI renderer (session/report.py) emits text; the dashboard needs structured
data. This module mirrors the same sections -- header, STRUCTURE, LEG ROLES, VWAP
MAP, KEY LEVELS, THE READ -- as nested dicts so the frontend can render them as
tables/components. Pure functions; no I/O.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..ml.signals import ScreenResult, Signal
from ..session.dissect import SessionDissection, group_vwap_events
from ..session.nested import NestedLeg
from ..session.report import render_read
from ..session.session import Session


def _hhmm(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")


def _leg_dict(leg: Any) -> dict[str, Any]:
    dur_min = int((leg.end_time - leg.start_time).total_seconds() / 60)
    return {
        "direction": leg.direction,
        "start_time": _hhmm(leg.start_time),
        "end_time": _hhmm(leg.end_time),
        "duration_min": dur_min,
        "start_price": round(float(leg.start_price), 2),
        "end_price": round(float(leg.end_price), 2),
        "magnitude": round(float(leg.magnitude), 2),
    }


def structure_to_list(nested: list[NestedLeg]) -> list[dict[str, Any]]:
    """STRUCTURE: the major skeleton, each leg with its nested sub-structure."""
    out = []
    for i, nl in enumerate(nested, 1):
        d = {"n": i, **_leg_dict(nl.major)}
        d["sub_scale"] = round(float(nl.sub_scale), 2) if nl.sub_legs else None
        d["sub_legs"] = [_leg_dict(s) for s in nl.sub_legs]
        out.append(d)
    return out


def leg_roles_to_list(d: SessionDissection) -> list[dict[str, Any]]:
    """LEG ROLES: each classified major leg with its role tags (one per skeleton
    leg -- this list length MUST equal structure_to_list's, the bug-#3 invariant)."""
    out = []
    for i, cl in enumerate(d.classified_legs, 1):
        out.append(
            {
                "n": i,
                "direction": cl.direction,
                "magnitude_atr": round(float(cl.magnitude_atr), 1),
                "vwap_start": cl.start_vs_vwap,
                "vwap_end": cl.end_vs_vwap,
                "roles": [r.value for r in cl.roles],
            }
        )
    return out


def vwap_map_to_list(d: SessionDissection) -> list[dict[str, Any]]:
    out = []
    for g in group_vwap_events(list(d.vwap_events)):
        out.append(
            {
                "type": g.type.value,
                "count": g.count,
                "first_time": _hhmm(g.first_time),
                "last_time": _hhmm(g.last_time),
                "outcome_atr": round(float(g.representative.outcome_atr), 2),
            }
        )
    return out


def levels_to_list(d: SessionDissection) -> list[dict[str, Any]]:
    return [
        {
            "time": _hhmm(e.time),
            "kind": e.kind.value,
            "level": round(float(e.level), 2),
            "held": bool(e.held),
            "outcome_atr": round(float(e.outcome_atr), 2),
        }
        for e in sorted(d.level_events, key=lambda x: x.time)
    ]


def dissection_to_dict(
    session: Session, d: SessionDissection, nested: list[NestedLeg]
) -> dict[str, Any]:
    """The full structured dissection: the report's sections as data."""
    rng = session.high - session.low
    return {
        "symbol": session.symbol,
        "date": pd.Timestamp(session.date).strftime("%Y-%m-%d"),
        "header": {
            "open": round(session.open_price, 2),
            "high": round(session.high, 2),
            "low": round(session.low, 2),
            "close": round(session.close_price, 2),
            "range": round(rng, 2),
            "range_pct": round(rng / session.open_price * 100, 2) if session.open_price else 0.0,
            "range_atr": round(float(d.summary.get("session_range_atr", float("nan"))), 1),
            "vwap_close": round(session.vwap_final, 2),
            "bias": str(d.summary.get("dominant_direction", "?")).upper(),
            "high_time": _hhmm(session.high_time),
            "low_time": _hhmm(session.low_time),
            "scale_atr": round(float(d.scale_atr), 2) if d.scale_atr == d.scale_atr else None,
        },
        "structure": structure_to_list(nested),
        "leg_roles": leg_roles_to_list(d),
        "vwap_map": vwap_map_to_list(d),
        "levels": levels_to_list(d),
        # render_read returns a titled section ("THE READ", rule, narrative); the
        # API wants only the narrative sentence (its last line).
        "read": (
            render_read(d, session).splitlines()[-1].strip()
            if render_read(d, session).strip()
            else ""
        ),
        "consistent": len(nested) == len(d.classified_legs),
    }


def signal_to_dict(s: Signal) -> dict[str, Any]:
    return s.as_dict()


def screen_to_dict(result: ScreenResult) -> dict[str, Any]:
    """The screen response: ranked validated signals + full transparency."""
    return {
        "summary": {
            "configs_evaluated": len(result.reports),
            "survived": len(result.survivors),
            "n_signals": len(result.signals),
        },
        "signals": [signal_to_dict(s) for s in result.signals],
        "reports": [
            {
                "symbol": r.symbol,
                "n_events": r.n_events,
                "n_signals": r.n_total_signals,
                "oos_edge_r": round(r.oos_net_expectancy_r, 4),
                "oos_auc": round(r.oos_auc, 4),
                "p_value": round(r.p_value, 4),
                "p_value_fdr": round(r.p_value_fdr, 4),
                "decay": round(r.decay, 4),
            }
            for r in sorted(result.reports, key=lambda x: x.p_value_fdr)
        ],
    }
