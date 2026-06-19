"""
Adversarial OFFLINE tests for the data layer.

NO network is touched here. Every test exercises pure logic:
  * endpoints.py  -- the registry, tier ordering, and gating set math.
  * client.py     -- URL building, cache-path determinism (same endpoint+params
                     -> same parquet path; different params -> different path),
                     apikey exclusion from the digest, and _cache_fresh TTL.
  * bars.py       -- Timeframe.minutes / .pandas_rule, resolve_timeframe, and
                     ResolutionGated being raised by fetch_intraday BEFORE any
                     network call when the pinned tier is too low.

FMPClient is always constructed with an EXPLICIT pinned tier so that no code
path calls .tier's lazy auto-detect (which would probe the network). .fetch()
is never called against the network. The only fetch_intraday calls are the
gated ones, which raise before the HTTP layer.
"""

from __future__ import annotations

import pytest

from engine.data.client import (
    AuthError,
    EndpointGated,
    FMPClient,
    FMPError,
    RateLimitExceeded,
)
from engine.data.endpoints import (
    BASE_URL,
    ENDPOINTS,
    Endpoint,
    Tier,
    available_endpoints,
    gated_endpoints,
    get_endpoint,
)
from engine.intraday.bars import (
    Resolution,
    ResolutionGated,
    Timeframe,
    fetch_intraday,
    resolve_timeframe,
    selectable_timeframes,
)


# ---------------------------------------------------------------------------
# Tier enum + ordering
# ---------------------------------------------------------------------------
def test_tier_ordering_is_ascending_capability():
    assert Tier.FREE < Tier.STARTER < Tier.PREMIUM < Tier.ULTIMATE
    assert int(Tier.FREE) == 0
    assert int(Tier.ULTIMATE) == 3


def test_tier_from_string_normalizes_case_and_whitespace():
    assert Tier.from_string("premium") is Tier.PREMIUM
    assert Tier.from_string("  Ultimate  ") is Tier.ULTIMATE
    assert Tier.from_string("FREE") is Tier.FREE


def test_tier_from_string_rejects_unknown():
    with pytest.raises(ValueError):
        Tier.from_string("GOLD")


# ---------------------------------------------------------------------------
# Endpoint registry + gating set math
# ---------------------------------------------------------------------------
def test_registry_keys_match_endpoint_keys():
    for key, ep in ENDPOINTS.items():
        assert ep.key == key
        assert isinstance(ep, Endpoint)


def test_get_endpoint_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_endpoint("does_not_exist")


def test_available_and_gated_partition_the_registry():
    """For every tier, available and gated must be disjoint and cover the whole
    registry -- no endpoint is both reachable and unreachable, none is dropped."""
    for tier in Tier:
        avail = set(available_endpoints(tier))
        gated = set(gated_endpoints(tier))
        assert avail.isdisjoint(gated)
        assert avail | gated == set(ENDPOINTS)


def test_gating_is_monotone_in_tier():
    """A higher tier can reach a superset of what a lower tier reaches."""
    prev = set()
    for tier in (Tier.FREE, Tier.STARTER, Tier.PREMIUM, Tier.ULTIMATE):
        avail = set(available_endpoints(tier))
        assert prev.issubset(avail)
        prev = avail


def test_low_tier_cannot_reach_ultimate_endpoint():
    """The headline gating contract: a FREE key cannot reach an Ultimate-gated
    endpoint, and the gated set reports it."""
    ult = get_endpoint("intraday_1min")
    assert ult.min_tier is Tier.ULTIMATE
    assert "intraday_1min" in gated_endpoints(Tier.FREE)
    assert "intraday_1min" in gated_endpoints(Tier.PREMIUM)
    assert "intraday_1min" not in available_endpoints(Tier.PREMIUM)


def test_high_tier_reaches_ultimate_endpoint():
    assert "intraday_1min" in available_endpoints(Tier.ULTIMATE)
    assert "institutional_13f" in available_endpoints(Tier.ULTIMATE)
    assert len(gated_endpoints(Tier.ULTIMATE)) == 0


def test_free_tier_reaches_only_free_endpoints():
    avail = available_endpoints(Tier.FREE)
    assert all(e.min_tier is Tier.FREE for e in avail.values())
    # The known FREE endpoints from the registry.
    assert "eod_light" in avail
    assert "profile" in avail
    # And nothing higher.
    assert "eod_full" not in avail  # STARTER
    assert "treasury_rates" not in avail  # PREMIUM


# ---------------------------------------------------------------------------
# FMPClient construction (offline, pinned tier)
# ---------------------------------------------------------------------------
def _client(tmp_path, tier=Tier.FREE):
    return FMPClient("test-key-123", tier=tier, cache_dir=tmp_path)


def test_client_rejects_empty_api_key(tmp_path):
    with pytest.raises(ValueError):
        FMPClient("", tier=Tier.FREE, cache_dir=tmp_path)


def test_client_rejects_whitespace_api_key(tmp_path):
    with pytest.raises(ValueError):
        FMPClient("   ", tier=Tier.FREE, cache_dir=tmp_path)


def test_client_pinned_tier_does_not_autodetect(tmp_path):
    """A pinned tier must be returned verbatim -- no network probe."""
    c = _client(tmp_path, Tier.PREMIUM)
    assert c.tier is Tier.PREMIUM


def test_client_tier_from_string(tmp_path):
    c = FMPClient("k", tier="ultimate", cache_dir=tmp_path)
    assert c.tier is Tier.ULTIMATE


# ---------------------------------------------------------------------------
# URL building (pure)
# ---------------------------------------------------------------------------
def test_build_url_includes_base_path_and_apikey(tmp_path):
    c = _client(tmp_path)
    url = c._build_url(get_endpoint("eod_light"), {"symbol": "TSLA"})
    assert url.startswith(BASE_URL + "/")
    assert "historical-price-eod/light" in url
    assert "symbol=TSLA" in url
    assert "apikey=test-key-123" in url


def test_build_url_drops_none_valued_params(tmp_path):
    c = _client(tmp_path)
    url = c._build_url(get_endpoint("eod_light"), {"symbol": "TSLA", "limit": None})
    assert "limit" not in url
    assert "symbol=TSLA" in url


# ---------------------------------------------------------------------------
# Cache-path determinism
# ---------------------------------------------------------------------------
def test_cache_path_is_deterministic_for_same_params(tmp_path):
    c = _client(tmp_path)
    ep = get_endpoint("intraday_5min")
    p1 = c._cache_path(ep, {"symbol": "TSLA", "limit": 5})
    p2 = c._cache_path(ep, {"symbol": "TSLA", "limit": 5})
    assert p1 == p2


def test_cache_path_is_order_independent(tmp_path):
    """Reordering params must not change the digest (sort_keys=True)."""
    c = _client(tmp_path)
    ep = get_endpoint("intraday_5min")
    p1 = c._cache_path(ep, {"symbol": "TSLA", "limit": 5})
    p2 = c._cache_path(ep, {"limit": 5, "symbol": "TSLA"})
    assert p1 == p2


def test_cache_path_differs_for_different_params(tmp_path):
    c = _client(tmp_path)
    ep = get_endpoint("intraday_5min")
    p_tsla = c._cache_path(ep, {"symbol": "TSLA", "limit": 5})
    p_aapl = c._cache_path(ep, {"symbol": "AAPL", "limit": 5})
    p_limit = c._cache_path(ep, {"symbol": "TSLA", "limit": 6})
    assert p_tsla != p_aapl
    assert p_tsla != p_limit


def test_cache_path_differs_for_different_endpoint(tmp_path):
    """Same params, different endpoint -> different parent dir AND path."""
    c = _client(tmp_path)
    params = {"symbol": "TSLA", "limit": 5}
    p5 = c._cache_path(get_endpoint("intraday_5min"), params)
    p15 = c._cache_path(get_endpoint("intraday_15min"), params)
    assert p5 != p15
    assert p5.parent.name == "intraday_5min"
    assert p15.parent.name == "intraday_15min"


def test_cache_path_excludes_apikey_from_digest(tmp_path):
    """The secret must never influence the cache key (it's stripped first)."""
    c = _client(tmp_path)
    ep = get_endpoint("intraday_5min")
    base = c._cache_path(ep, {"symbol": "TSLA", "limit": 5})
    with_key = c._cache_path(ep, {"symbol": "TSLA", "limit": 5, "apikey": "SECRET"})
    assert base == with_key


def test_cache_path_shape(tmp_path):
    c = _client(tmp_path)
    ep = get_endpoint("intraday_5min")
    p = c._cache_path(ep, {"symbol": "TSLA"})
    assert p.suffix == ".parquet"
    assert len(p.stem) == 16  # sha256 truncated to 16 hex chars
    assert p.parent.name == ep.key
    assert p.parent.exists()  # _cache_path mkdirs the subdir


# ---------------------------------------------------------------------------
# _cache_fresh TTL logic
# ---------------------------------------------------------------------------
def test_cache_fresh_false_when_missing(tmp_path):
    c = _client(tmp_path)
    p = c._cache_path(get_endpoint("intraday_5min"), {"symbol": "TSLA"})
    assert c._cache_fresh(p) is False


def test_cache_fresh_true_when_ttl_none_and_exists(tmp_path):
    """ttl None means immutable cache: any existing file is fresh forever."""
    c = FMPClient("k", tier=Tier.FREE, cache_dir=tmp_path, cache_ttl_days=None)
    p = c._cache_path(get_endpoint("intraday_5min"), {"symbol": "TSLA"})
    p.write_bytes(b"data")
    assert c._cache_fresh(p) is True


def test_cache_fresh_respects_positive_ttl(tmp_path):
    """A file just written is within a generous TTL -> fresh."""
    c = FMPClient("k", tier=Tier.FREE, cache_dir=tmp_path, cache_ttl_days=7.0)
    p = c._cache_path(get_endpoint("intraday_5min"), {"symbol": "TSLA"})
    p.write_bytes(b"data")
    assert c._cache_fresh(p) is True


def test_cache_fresh_stale_when_older_than_ttl(tmp_path):
    """Backdate mtime well beyond the TTL -> stale (must refetch)."""
    import os
    import time

    c = FMPClient("k", tier=Tier.FREE, cache_dir=tmp_path, cache_ttl_days=1.0)
    p = c._cache_path(get_endpoint("intraday_5min"), {"symbol": "TSLA"})
    p.write_bytes(b"data")
    old = time.time() - 5 * 86400.0  # 5 days old, TTL is 1 day
    os.utime(p, (old, old))
    assert c._cache_fresh(p) is False


def test_cache_fresh_zero_ttl_treats_just_written_as_stale(tmp_path):
    """ttl=0 means 'don't trust the cache': even a brand-new file is stale,
    because its age (>0) exceeds the 0-day window. Locked as intended."""
    c = FMPClient("k", tier=Tier.FREE, cache_dir=tmp_path, cache_ttl_days=0.0)
    p = c._cache_path(get_endpoint("intraday_5min"), {"symbol": "TSLA"})
    p.write_bytes(b"data")
    assert c._cache_fresh(p) is False


# ---------------------------------------------------------------------------
# Error type hierarchy
# ---------------------------------------------------------------------------
def test_error_hierarchy():
    assert issubclass(AuthError, FMPError)
    assert issubclass(EndpointGated, FMPError)
    assert issubclass(RateLimitExceeded, FMPError)
    assert issubclass(FMPError, RuntimeError)


def test_endpoint_gated_carries_endpoint_and_tier():
    eg = EndpointGated(get_endpoint("intraday_1min"), Tier.FREE)
    assert eg.endpoint.key == "intraday_1min"
    assert eg.tier is Tier.FREE
    assert "ULTIMATE" in str(eg)
    assert "FREE" in str(eg)


# ---------------------------------------------------------------------------
# Timeframe enum: minutes + pandas_rule mapping
# ---------------------------------------------------------------------------
def test_timeframe_minutes_mapping():
    expected = {
        Timeframe.M1: 1,
        Timeframe.M2: 2,
        Timeframe.M3: 3,
        Timeframe.M4: 4,
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.M30: 30,
        Timeframe.H1: 60,
        Timeframe.H2: 120,
        Timeframe.H4: 240,
    }
    for tf, mins in expected.items():
        assert tf.minutes == mins


def test_timeframe_pandas_rule_is_minutes_suffixed():
    assert Timeframe.M1.pandas_rule == "1min"
    assert Timeframe.M5.pandas_rule == "5min"
    assert Timeframe.H1.pandas_rule == "60min"
    assert Timeframe.H4.pandas_rule == "240min"
    # Rule must always be derived from minutes, never the enum value.
    for tf in Timeframe:
        assert tf.pandas_rule == f"{tf.minutes}min"


# ---------------------------------------------------------------------------
# resolve_timeframe: native vs resample + min_tier
# ---------------------------------------------------------------------------
def test_resolve_native_timeframes_need_no_resample():
    for tf in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1):
        r = resolve_timeframe(tf)
        assert isinstance(r, Resolution)
        assert r.needs_resample is False
        assert r.native_source is tf


def test_resolve_native_min_tiers():
    assert resolve_timeframe(Timeframe.M1).min_tier is Tier.ULTIMATE
    assert resolve_timeframe(Timeframe.M5).min_tier is Tier.PREMIUM
    assert resolve_timeframe(Timeframe.H1).min_tier is Tier.PREMIUM


def test_resolve_resampled_timeframes_inherit_source_tier():
    """2/3/4-min resample from 1-min -> inherit ULTIMATE; 2/4-hour from
    1-hour -> inherit PREMIUM."""
    for tf in (Timeframe.M2, Timeframe.M3, Timeframe.M4):
        r = resolve_timeframe(tf)
        assert r.needs_resample is True
        assert r.native_source is Timeframe.M1
        assert r.min_tier is Tier.ULTIMATE
    for tf in (Timeframe.H2, Timeframe.H4):
        r = resolve_timeframe(tf)
        assert r.needs_resample is True
        assert r.native_source is Timeframe.H1
        assert r.min_tier is Tier.PREMIUM


# ---------------------------------------------------------------------------
# selectable_timeframes: per-tier availability
# ---------------------------------------------------------------------------
def test_selectable_free_reaches_nothing_intraday():
    sel = selectable_timeframes(Tier.FREE)
    assert set(sel) == {tf.value for tf in Timeframe}
    assert all(v["available"] is False for v in sel.values())


def test_selectable_premium_reaches_5min_but_not_1min():
    sel = selectable_timeframes(Tier.PREMIUM)
    assert sel["5min"]["available"] is True
    assert sel["1hour"]["available"] is True
    assert sel["2hour"]["available"] is True  # resampled from 1hour
    assert sel["1min"]["available"] is False  # Ultimate
    assert sel["3min"]["available"] is False  # resampled from 1min -> Ultimate


def test_selectable_ultimate_reaches_everything():
    sel = selectable_timeframes(Tier.ULTIMATE)
    assert all(v["available"] is True for v in sel.values())


# ---------------------------------------------------------------------------
# fetch_intraday gating raises BEFORE any network (pinned low tier)
# ---------------------------------------------------------------------------
def test_fetch_intraday_gated_raises_resolution_gated(tmp_path):
    """A FREE-pinned client requesting M1 (Ultimate) must raise ResolutionGated
    before touching the network -- the raise short-circuits client.fetch()."""
    c = FMPClient("k", tier=Tier.FREE, cache_dir=tmp_path)
    with pytest.raises(ResolutionGated) as exc:
        fetch_intraday(c, "TSLA", Timeframe.M1)
    assert exc.value.tier is Tier.FREE


def test_fetch_intraday_resampled_gated_names_native_dependency(tmp_path):
    """A PREMIUM-pinned client requesting M3 (resample from 1-min/Ultimate) must
    raise ResolutionGated naming the 1-min native dependency, not 3-min."""
    c = FMPClient("k", tier=Tier.PREMIUM, cache_dir=tmp_path)
    with pytest.raises(ResolutionGated) as exc:
        fetch_intraday(c, "TSLA", Timeframe.M3)
    assert exc.value.endpoint.key == "intraday_1min"
    assert exc.value.tier is Tier.PREMIUM


def test_resolution_gated_is_endpoint_gated_subclass():
    assert issubclass(ResolutionGated, EndpointGated)
