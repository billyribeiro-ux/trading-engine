"""
Professional session report renderer.

Turns a SessionDissection + nested structure into a clean, layered, desk-grade
report instead of a flat event dump. Layout:

    HEADER     -- symbol, date, O/H/L/C, range, VWAP, regime one-liner
    SKELETON   -- the day's major legs (the 4-6 leg story), each with role tags
                  and, nested beneath, its internal sub-structure
    VWAP MAP   -- grouped VWAP interaction events (reclaims/losses/retests),
                  de-duplicated, with outcomes
    LEVELS     -- structural level tests (prior-day, opening range) held/broke
    READ       -- a plain-language narrative of the session

Everything is measured; the narrative is assembled from the measurements, not
hand-written commentary. Pure functions returning strings, so the same renderer
serves the CLI, a file, or a future UI.
"""

from __future__ import annotations

import pandas as pd

from .dissect import (
    LegRole,
    SessionDissection,
    VwapEventType,
    group_vwap_events,
)
from .nested import NestedLeg
from .session import Session

_RULE = "─" * 78
_THIN = "·" * 78


def _fmt_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def _role_tags(roles: tuple[LegRole, ...]) -> str:
    pretty = {
        LegRole.FLUSH: "FLUSH",
        LegRole.REVERSAL: "REVERSAL",
        LegRole.VWAP_RECLAIM: "VWAP reclaim",
        LegRole.VWAP_LOSS: "VWAP loss",
        LegRole.TREND_LEG: "trend",
        LegRole.RETRACE: "retrace",
        LegRole.HOD_TEST: "HOD test",
        LegRole.LOD_TEST: "LOD test",
        LegRole.FADE: "fade",
        LegRole.UNCLASSIFIED: "—",
    }
    return ", ".join(pretty.get(r, r.value) for r in roles)


def _arrow(direction: str) -> str:
    return "▲" if direction == "up" else "▼"


def render_header(session: Session, d: SessionDissection) -> str:
    rng = session.high - session.low
    rng_pct = rng / session.open_price * 100 if session.open_price else 0.0
    dom = d.summary.get("dominant_direction", "?")
    range_atr = d.summary.get("session_range_atr", float("nan"))
    lines = [
        _RULE,
        f"  {session.symbol}   {session.date.strftime('%A, %B %-d, %Y')}",
        _RULE,
        f"  Open {session.open_price:8.2f}    High {session.high:8.2f}    "
        f"Low {session.low:8.2f}    Close {session.close_price:8.2f}",
        f"  Range {rng:7.2f} ({rng_pct:+.2f}%, {range_atr:.1f} ATR)    "
        f"VWAP close {session.vwap_final:8.2f}    Bias {dom.upper()}",
        f"  Session low {session.low:.2f} at {_fmt_time(session.low_time)}    "
        f"Session high {session.high:.2f} at {_fmt_time(session.high_time)}",
    ]
    return "\n".join(lines)


def render_skeleton(nested: list[NestedLeg], session: Session) -> str:
    lines = ["", "  STRUCTURE  (the day's skeleton — major legs)", _THIN]
    if not nested:
        lines.append("  (no major structural legs detected)")
        return "\n".join(lines)

    for i, nl in enumerate(nested, 1):
        mleg = nl.major
        dur_min = int((mleg.end_time - mleg.start_time).total_seconds() / 60)
        # Major leg line.
        lines.append(
            f"  {i}. {_arrow(mleg.direction)} {mleg.direction.upper():4} "
            f"{_fmt_time(mleg.start_time)}→{_fmt_time(mleg.end_time)} "
            f"({dur_min:>3}m)   {mleg.start_price:8.2f} → {mleg.end_price:8.2f}   "
            f"{mleg.magnitude:6.2f} pts"
        )
        # Nested sub-structure, if the major leg has internal swings.
        if nl.sub_legs:
            lines.append(f"      └─ internal: {nl.n_sub} swings (scale {nl.sub_scale:.2f} ATR)")
            for sub in nl.sub_legs:
                sdur = int((sub.end_time - sub.start_time).total_seconds() / 60)
                lines.append(
                    f"         {_arrow(sub.direction)} "
                    f"{_fmt_time(sub.start_time)}→{_fmt_time(sub.end_time)} "
                    f"({sdur:>3}m)  {sub.start_price:8.2f}→{sub.end_price:8.2f}  "
                    f"{sub.magnitude:5.2f} pts"
                )
    return "\n".join(lines)


def render_leg_roles(d: SessionDissection) -> str:
    lines = ["", "  LEG ROLES  (what each major move was)", _THIN]
    for i, cl in enumerate(d.classified_legs, 1):
        tags = _role_tags(cl.roles)
        lines.append(
            f"  {i}. {_arrow(cl.direction)} {cl.magnitude_atr:4.1f} ATR   "
            f"VWAP {cl.start_vs_vwap}→{cl.end_vs_vwap}   [{tags}]"
        )
    return "\n".join(lines)


def render_vwap_map(d: SessionDissection) -> str:
    groups = group_vwap_events(list(d.vwap_events))
    lines = ["", "  VWAP MAP  (interactions with the volume-weighted average)", _THIN]
    if not groups:
        lines.append("  (no VWAP interactions)")
        return "\n".join(lines)
    for g in groups:
        marker = {
            VwapEventType.RECLAIM: "↑ reclaim",
            VwapEventType.LOSS: "↓ loss",
            VwapEventType.RETEST_HOLD: "● held",
            VwapEventType.RETEST_FAIL: "○ failed",
        }.get(g.type, g.type.value)
        if g.count == 1:
            span = _fmt_time(g.first_time)
        else:
            span = f"{_fmt_time(g.first_time)}–{_fmt_time(g.last_time)} ×{g.count}"
        lines.append(
            f"  {span:<16} {marker:<12} → {g.representative.outcome_atr:+.2f} ATR follow-through"
        )
    return "\n".join(lines)


def render_levels(d: SessionDissection) -> str:
    lines = ["", "  KEY LEVELS  (structural references tested)", _THIN]
    if not d.level_events:
        lines.append("  (no level tests)")
        return "\n".join(lines)
    pretty = {
        "prior_high": "Prior-day high",
        "prior_low": "Prior-day low",
        "prior_close": "Prior close",
        "open_range_high": "Opening-range high",
        "open_range_low": "Opening-range low",
        "session_high": "Session high",
        "session_low": "Session low",
    }
    for e in sorted(d.level_events, key=lambda x: x.time):
        verdict = "HELD " if e.held else "BROKE"
        name = pretty.get(e.kind.value, e.kind.value)
        lines.append(
            f"  {_fmt_time(e.time)}  {name:<20} @ {e.level:8.2f}   "
            f"{verdict}  → {e.outcome_atr:+.2f} ATR"
        )
    return "\n".join(lines)


def render_read(d: SessionDissection, session: Session) -> str:
    """Plain-language narrative assembled from the measurements."""
    s = d.summary
    legs = d.classified_legs
    if not legs:
        return ""

    parts: list[str] = []
    # Opening behaviour: detect the move from the open to the session extreme,
    # regardless of how many legs it spans. The first major extreme reached --
    # session low (sold off) or session high (ran) -- defines the open.
    low_first = session.low_time <= session.high_time
    if low_first:
        # Opened and sold off to the session low.
        drop_pts = session.open_price - session.low
        drop_atr = drop_pts / session.atr_mean if session.atr_mean > 0 else 0.0
        parts.append(
            f"Opened at {session.open_price:.2f} and sold off "
            f"{drop_atr:.1f} ATR into the session low at {session.low:.2f} "
            f"({_fmt_time(session.low_time)})."
        )
    else:
        run_pts = session.high - session.open_price
        run_atr = run_pts / session.atr_mean if session.atr_mean > 0 else 0.0
        parts.append(
            f"Opened at {session.open_price:.2f} and ran "
            f"{run_atr:.1f} ATR to the session high at {session.high:.2f} "
            f"({_fmt_time(session.high_time)})."
        )

    # Reversal + reclaim.
    reclaim_leg = next((cl for cl in legs if LegRole.VWAP_RECLAIM in cl.roles), None)
    if reclaim_leg is not None:
        parts.append(
            f"Reversed and reclaimed VWAP, then "
            f"{'extended toward the high' if reclaim_leg.direction == 'up' else 'pressed toward the low'}."
        )

    # HOD/LOD resolution: judge by where the day actually RESOLVED (close
    # relative to the extremes), not a transient touch mid-session.
    close_to_high = abs(session.close_price - session.high)
    close_to_low = abs(session.close_price - session.low)
    closed_strong = close_to_high < close_to_low  # closed nearer the high

    if closed_strong and s.get("dominant_direction") == "up":
        parts.append(
            f"Trended up to test the high of day ({session.high:.2f}), "
            f"closing strong at {session.close_price:.2f}."
        )
    elif not closed_strong and s.get("dominant_direction") == "down":
        parts.append(
            f"Pressed down to the low ({session.low:.2f}), "
            f"closing weak at {session.close_price:.2f}."
        )
    else:
        rng = session.high - session.low
        pos = (session.close_price - session.low) / rng if rng > 0 else 0.5
        zone = "upper" if pos > 0.66 else ("lower" if pos < 0.33 else "middle")
        parts.append(f"Closed in the {zone} of the range at {session.close_price:.2f}.")

    # VWAP character.
    holds = s.get("n_retest_holds", 0)
    fails = s.get("n_retest_fails", 0)
    if holds > fails and holds > 0:
        parts.append("VWAP acted as support — retests largely held.")
    elif fails > holds and fails > 0:
        parts.append("VWAP was contested — multiple retests failed before resolution.")

    lines = ["", "  THE READ", _THIN, "  " + " ".join(parts)]
    return "\n".join(lines)


def render_report(
    session: Session,
    d: SessionDissection,
    nested: list[NestedLeg],
) -> str:
    """Assemble the full professional report."""
    sections = [
        render_header(session, d),
        render_skeleton(nested, session),
        render_leg_roles(d),
        render_vwap_map(d),
        render_levels(d),
        render_read(d, session),
        _RULE,
    ]
    return "\n".join(sections)
