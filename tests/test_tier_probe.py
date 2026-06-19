"""
Offline regression lock for the FMP tier-detection probe (engine bug found
2026-06-18).

The Ultimate tier is detected by probing the 13F institutional-ownership
endpoint. FMP migrated 13F to the /stable API; the old 'institutional-ownership/
holdings' path now 404s. Because _detect_tier treated any non-200 as "no access,"
a real ULTIMATE key was silently downgraded to PREMIUM, which then blocked 1min
intraday data the key actually had.

This locks the corrected probe so the stale path can never silently return. It is
fully offline -- it asserts on the endpoint REGISTRY, not the network.
"""

from __future__ import annotations

from engine.data.endpoints import ENDPOINTS, Tier


def test_ultimate_probe_uses_live_stable_path_not_stale_404_path():
    ep = ENDPOINTS["institutional_13f"]
    assert ep.path == "institutional-ownership/latest", (
        "tier-probe path regressed; 'institutional-ownership/holdings' 404s on the "
        "stable API and downgrades Ultimate keys to Premium"
    )
    assert ep.path != "institutional-ownership/holdings"
    assert ep.min_tier == Tier.ULTIMATE
    # The 'latest' endpoint is market-wide (page/limit), not symbol-scoped.
    assert ep.symbol_param is None
