"""
FMP HTTP client: tier-aware, rate-limited, disk-cached.

Design goals
------------
* Never spend a request on a tier-gated endpoint -> raise EndpointGated so
  callers can record the signal as UNAVAILABLE (honest absence, not fake zero).
* Respect the per-minute budget with a token-bucket limiter; FREE also honours
  a hard daily cap.
* Cache every successful response to Parquet keyed by (endpoint, params). Price
  history is immutable once the session closes, so cached reads cost nothing and
  keep us well under the 50GB/30d Premium bandwidth ceiling.
* Retry transient failures (429, 5xx, connection) with exponential backoff and
  jitter. Surface auth/permission errors immediately -- retrying a 401 is waste.
* Detect the tier at runtime by probing a known Premium-gated endpoint, unless
  the caller pins it explicitly.

This module has no engine-specific knowledge. It returns pandas DataFrames with
parsed datetimes and nothing else. Feature/attribution logic lives downstream.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
import requests

from .endpoints import (
    BASE_URL,
    FREE_DAILY_CAP,
    RATE_LIMIT_PER_MIN,
    Endpoint,
    Tier,
    available_endpoints,
    gated_endpoints,
    get_endpoint,
)

logger = logging.getLogger("engine.data.client")

_DEFAULT_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "fmp_engine"
_TIMEOUT: Final[float] = 30.0
_MAX_RETRIES: Final[int] = 5
_BACKOFF_BASE: Final[float] = 0.75  # seconds; grows 0.75,1.5,3,6,12 (+jitter)
_TIER_PROBE_KEY: Final[str] = "treasury_rates"  # Premium-gated, symbol-free, cheap


class FMPError(RuntimeError):
    """Base class for all client-raised errors."""


class AuthError(FMPError):
    """401/403 -- bad key or insufficient permission. Not retried."""


class EndpointGated(FMPError):
    """Active tier is below the endpoint's minimum tier. Not a network error."""

    def __init__(self, endpoint: Endpoint, tier: Tier) -> None:
        self.endpoint = endpoint
        self.tier = tier
        super().__init__(
            f"Endpoint {endpoint.key!r} requires {endpoint.min_tier.name}; "
            f"active tier is {tier.name}. Data recorded as UNAVAILABLE."
        )


class RateLimitExceeded(FMPError):
    """Daily cap hit on FREE, or limiter could not satisfy the request."""


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (thread-safe).
# ---------------------------------------------------------------------------
class _RateLimiter:
    """
    Sliding-window limiter. Records timestamps of the last N requests and
    blocks until the oldest falls outside the 60s window when the budget is
    exhausted. Simpler than a leaky bucket and exact for a per-minute cap.
    """

    def __init__(self, per_minute: int) -> None:
        self._per_minute = max(1, per_minute)
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._events and self._events[0] < cutoff:
                    self._events.popleft()
                if len(self._events) < self._per_minute:
                    self._events.append(now)
                    return
                sleep_for = self._events[0] + 60.0 - now
            # Sleep outside the lock so other threads can drain.
            time.sleep(max(0.01, sleep_for))


@dataclass
class _DailyCounter:
    """Hard daily-request cap for FREE tier. Resets at UTC midnight."""

    cap: int
    _count: int = 0
    _day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_increment(self) -> None:
        with self._lock:
            today = datetime.now(timezone.utc).date()
            if today != self._day:
                self._day = today
                self._count = 0
            if self._count >= self.cap:
                raise RateLimitExceeded(
                    f"FREE daily cap of {self.cap} requests reached for {today}."
                )
            self._count += 1


class FMPClient:
    """
    Thread-safe FMP /stable/ client.

    Parameters
    ----------
    api_key : str
        FMP API key.
    tier : Tier | str | None
        Pin the tier, or None to auto-detect at first use.
    cache_dir : Path | None
        Parquet cache root. None -> ~/.cache/fmp_engine.
    cache_ttl_days : float | None
        If set, cached files older than this are refetched. None -> immutable
        (price history never changes post-session; default for backtests).
    session : requests.Session | None
        Inject for testing.
    """

    def __init__(
        self,
        api_key: str,
        tier: Tier | str | None = None,
        cache_dir: Path | None = None,
        cache_ttl_days: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")
        self._api_key = api_key.strip()
        self._cache_dir = (cache_dir or _DEFAULT_CACHE_DIR).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_days = cache_ttl_days
        self._session = session or requests.Session()

        if tier is None:
            self._tier: Tier | None = None  # lazy-detected
        elif isinstance(tier, str):
            self._tier = Tier.from_string(tier)
        else:
            self._tier = tier

        # Limiter sized to the pinned tier, or to PREMIUM as a safe default
        # until detection completes (detection itself makes one request).
        budget_tier = self._tier if self._tier is not None else Tier.PREMIUM
        self._limiter = _RateLimiter(RATE_LIMIT_PER_MIN[budget_tier])
        self._daily = _DailyCounter(FREE_DAILY_CAP)

    # -- tier ---------------------------------------------------------------
    @property
    def tier(self) -> Tier:
        if self._tier is None:
            self._tier = self._detect_tier()
            self._limiter = _RateLimiter(RATE_LIMIT_PER_MIN[self._tier])
            logger.info("Detected FMP tier: %s", self._tier.name)
        return self._tier

    def _detect_tier(self) -> Tier:
        """
        Probe upward. We attempt the Premium-gated treasury endpoint and an
        Ultimate-gated 13F endpoint to place the tier. A 200 means access; a
        403/401 on a specific endpoint means it's above our tier.
        """
        # Probe Ultimate first (13F). If it works, we're Ultimate.
        if self._probe("institutional_13f", {"page": 0, "limit": 1}):
            return Tier.ULTIMATE
        # Probe Premium (treasury). If it works, we're Premium.
        if self._probe(_TIER_PROBE_KEY, {}):
            return Tier.PREMIUM
        # Probe Starter (full EOD). If it works, we're Starter.
        if self._probe("eod_full", {"symbol": "AAPL", "limit": 1}):
            return Tier.STARTER
        return Tier.FREE

    def _probe(self, endpoint_key: str, params: dict[str, Any]) -> bool:
        ep = get_endpoint(endpoint_key)
        url = self._build_url(ep, params)
        try:
            self._limiter.acquire()
            resp = self._session.get(url, timeout=_TIMEOUT)
        except requests.RequestException:
            return False
        if resp.status_code in (401, 403):
            return False
        return resp.status_code == 200

    # -- capability report --------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        """What this key can and cannot reach. Surfaced in the run header."""
        t = self.tier
        return {
            "tier": t.name,
            "rate_limit_per_min": RATE_LIMIT_PER_MIN[t],
            "available": sorted(available_endpoints(t)),
            "unavailable": {k: e.min_tier.name for k, e in gated_endpoints(t).items()},
        }

    # -- URL + cache plumbing ----------------------------------------------
    def _build_url(self, ep: Endpoint, params: dict[str, Any]) -> str:
        q = {k: v for k, v in params.items() if v is not None}
        q["apikey"] = self._api_key
        query = "&".join(f"{k}={v}" for k, v in q.items())
        return f"{BASE_URL}/{ep.path}?{query}"

    def _cache_path(self, ep: Endpoint, params: dict[str, Any]) -> Path:
        safe = {k: v for k, v in params.items() if k != "apikey"}
        digest = hashlib.sha256(json.dumps(safe, sort_keys=True, default=str).encode()).hexdigest()[
            :16
        ]
        sub = self._cache_dir / ep.key
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{digest}.parquet"

    def _cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self._cache_ttl_days is None:
            return True
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        return age_days <= self._cache_ttl_days

    # -- core request -------------------------------------------------------
    def fetch(
        self,
        endpoint_key: str,
        *,
        symbol: str | None = None,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch one endpoint as a DataFrame.

        Raises
        ------
        EndpointGated   if the active tier cannot reach the endpoint.
        AuthError       on 401/403 (after tier check passes).
        RateLimitExceeded on FREE daily-cap exhaustion.
        FMPError        on persistent transient failure.
        """
        ep = get_endpoint(endpoint_key)
        if self.tier < ep.min_tier:
            raise EndpointGated(ep, self.tier)

        call_params: dict[str, Any] = dict(params or {})
        if symbol is not None:
            if ep.symbol_param is None:
                logger.debug(
                    "Endpoint %s is market-wide; ignoring symbol=%s for the "
                    "request and filtering locally if needed.",
                    ep.key,
                    symbol,
                )
            else:
                call_params[ep.symbol_param] = symbol

        cache_path = self._cache_path(ep, call_params)
        if use_cache and self._cache_fresh(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:  # corrupt cache -> refetch
                logger.warning("Corrupt cache at %s; refetching.", cache_path)

        payload = self._request_with_retry(ep, call_params)
        df = self._to_frame(payload, ep)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:  # caching is best-effort, never fatal
            logger.warning("Failed to cache %s: %s", cache_path, exc)
        return df

    def _request_with_retry(self, ep: Endpoint, params: dict[str, Any]) -> list | dict:
        url = self._build_url(ep, params)
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            if self.tier == Tier.FREE:
                self._daily.check_and_increment()
            self._limiter.acquire()
            try:
                resp = self._session.get(url, timeout=_TIMEOUT)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise FMPError(f"Non-JSON 200 from {ep.key}: {resp.text[:200]!r}") from exc

            if resp.status_code in (401, 403):
                raise AuthError(
                    f"{resp.status_code} on {ep.key}. Check API key / plan. "
                    f"Body: {resp.text[:200]!r}"
                )

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_exc = FMPError(f"{resp.status_code} on {ep.key}: {resp.text[:200]!r}")
                self._sleep_backoff(attempt, retry_after=resp.headers.get("Retry-After"))
                continue

            # Other 4xx: not retryable.
            raise FMPError(f"{resp.status_code} on {ep.key}: {resp.text[:200]!r}")

        raise FMPError(f"Exhausted {_MAX_RETRIES} retries on {ep.key}. Last error: {last_exc}")

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: str | None = None) -> None:
        if retry_after is not None:
            try:
                time.sleep(min(60.0, float(retry_after)))
                return
            except (TypeError, ValueError):
                pass
        # Exponential backoff with full jitter.
        import random

        ceiling = _BACKOFF_BASE * (2**attempt)
        time.sleep(random.uniform(0.0, ceiling))

    @staticmethod
    def _to_frame(payload: list | dict, ep: Endpoint) -> pd.DataFrame:
        if isinstance(payload, dict):
            # Some endpoints wrap rows under a key; some return a single object.
            for candidate in ("historical", "results", "data"):
                if candidate in payload and isinstance(payload[candidate], list):
                    payload = payload[candidate]
                    break
            else:
                payload = [payload]
        if not payload:
            return pd.DataFrame()

        df = pd.DataFrame(payload)
        # Normalise common datetime columns so downstream code is uniform.
        for col in ("date", "datetime", "publishedDate", "acceptedDate", "fillingDate"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
        # Sort price-like frames ascending by their time column.
        time_col = next((c for c in ("date", "datetime") if c in df.columns), None)
        if time_col is not None:
            df = df.sort_values(time_col).reset_index(drop=True)
        return df
