"""
Network-gated integration smoke test.

This file hits the LIVE FMP API and is SKIPPED unless FMP_API_KEY is set in the
environment. In CI without a key (the default, and the correct posture) it must
report SKIPPED -- never error, never silently pass.

What it asserts when a key IS present:
  * dissect_real_session(client, "TSLA", Timeframe.M1) returns a coherent
    (session, dissection, nested) triple.
  * STRUCTURE == LEG ROLES: the nested structure has exactly one entry per
    classified leg -- len(nested) == len(dissection.classified_legs).

The structural invariant is the real check; the live fetch is the integration
smoke. We pin no live prices (non-deterministic), only relationships that must
hold for ANY real session.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not os.environ.get("FMP_API_KEY"),
        reason="no FMP_API_KEY",
    ),
]


def test_dissect_real_tsla_session_structure_matches_leg_roles():
    """Live TSLA M1 dissection: the returned triple is coherent and the nested
    structure aligns 1:1 with the classified legs.

    1min is Ultimate-gated. If the active key's tier can't reach it, SKIP with the
    detected tier in the message (rather than hard-fail on a tier limitation)."""
    from engine.data.client import FMPClient
    from engine.intraday.bars import Timeframe, resolve_timeframe
    from engine.session.runner import dissect_real_session

    client = FMPClient(os.environ["FMP_API_KEY"])
    if client.tier < resolve_timeframe(Timeframe.M1).min_tier:
        pytest.skip(f"key tier {client.tier.name} cannot reach 1min (Ultimate-gated)")
    session, dissection, nested = dissect_real_session(client, "TSLA", Timeframe.M1)

    # The triple is coherent.
    assert session is not None
    assert dissection is not None
    assert dissection.symbol == "TSLA"
    assert isinstance(nested, list)

    # Legs were actually classified on a real session.
    assert len(dissection.classified_legs) > 0

    # STRUCTURE == LEG ROLES: one nested entry per classified leg.
    assert len(nested) == len(dissection.classified_legs)

    # scale_atr is a real (non-NaN) magnitude when legs exist.
    assert dissection.scale_atr == dissection.scale_atr  # not NaN
    assert dissection.scale_atr > 0
