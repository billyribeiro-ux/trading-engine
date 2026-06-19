"""
FMP /stable/ endpoint registry with tier gating.

Verified against Financial Modeling Prep documentation as of 2026-06-17.

Tier hierarchy (ascending capability):
    FREE     -> 250 req/day,  EOD only, ~5yr history, 500MB/30d
    STARTER  -> 300 req/min,  ~5yr history, US, annual fundamentals
    PREMIUM  -> 750 req/min,  30yr history, intraday charts, indicators,
                calendars, ratings, insider, estimates, sentiment, news
    ULTIMATE -> 3000 req/min, + 13F holdings, ETF/MF holdings, transcripts,
                1-min intraday full depth, bulk/batch, global coverage

The registry lets the client (a) build correct /stable/ URLs and (b) refuse
to spend a request on an endpoint the active tier cannot access, marking the
corresponding data as UNAVAILABLE rather than silently returning empty.

Nothing here fabricates data. If an endpoint is gated above the active tier,
the attribution layer treats its signal as absent, never inferred-as-present.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class Tier(IntEnum):
    """Ordered so that `active_tier >= endpoint.min_tier` is a valid gate check."""

    FREE = 0
    STARTER = 1
    PREMIUM = 2
    ULTIMATE = 3

    @classmethod
    def from_string(cls, name: str) -> Tier:
        key = name.strip().upper()
        if key not in cls.__members__:
            raise ValueError(
                f"Unknown FMP tier {name!r}. Expected one of {', '.join(cls.__members__)}."
            )
        return cls[key]


# Requests-per-minute budget per tier. FREE is per-day (250); we model it as a
# conservative per-minute trickle so the limiter logic stays uniform, and the
# client additionally enforces the daily cap for FREE.
RATE_LIMIT_PER_MIN: Final[dict[Tier, int]] = {
    Tier.FREE: 4,  # ~250/day spread; daily cap enforced separately
    Tier.STARTER: 300,
    Tier.PREMIUM: 750,
    Tier.ULTIMATE: 3000,
}

FREE_DAILY_CAP: Final[int] = 250

BASE_URL: Final[str] = "https://financialmodelingprep.com/stable"


@dataclass(frozen=True)
class Endpoint:
    """
    A single FMP /stable/ endpoint.

    path:      URL path under BASE_URL (no leading slash).
    min_tier:  lowest tier that can access it.
    symbol_param: query-param name carrying the ticker, or None for
                  market-wide endpoints (calendars, treasury, etc.).
    interval_path: if True, the resolution is a path segment
                   (historical-chart/{interval}) rather than a query param.
    description: human-readable purpose, surfaced in capability reports.
    """

    key: str
    path: str
    min_tier: Tier
    symbol_param: str | None = "symbol"
    interval_path: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Registry. Keys are stable identifiers used throughout the engine.
# ---------------------------------------------------------------------------
_ENDPOINTS: Final[tuple[Endpoint, ...]] = (
    # --- Price / volume -----------------------------------------------------
    Endpoint(
        key="eod_full",
        path="historical-price-eod/full",
        min_tier=Tier.STARTER,
        description="Daily OHLCV + VWAP, split/div adjusted. Swing/long-term backbone (30yr on Premium+).",
    ),
    Endpoint(
        key="eod_light",
        path="historical-price-eod/light",
        min_tier=Tier.FREE,
        description="Daily close + volume only. Fallback when full EOD is gated.",
    ),
    Endpoint(
        key="intraday_1min",
        path="historical-chart/1min",
        min_tier=Tier.ULTIMATE,
        interval_path=True,
        description="1-minute OHLCV. Full depth Ultimate-only. Scalp layer.",
    ),
    Endpoint(
        key="intraday_5min",
        path="historical-chart/5min",
        min_tier=Tier.PREMIUM,
        interval_path=True,
        description="5-minute OHLCV. Day-trade layer.",
    ),
    Endpoint(
        key="intraday_15min",
        path="historical-chart/15min",
        min_tier=Tier.PREMIUM,
        interval_path=True,
        description="15-minute OHLCV.",
    ),
    Endpoint(
        key="intraday_30min",
        path="historical-chart/30min",
        min_tier=Tier.PREMIUM,
        interval_path=True,
        description="30-minute OHLCV.",
    ),
    Endpoint(
        key="intraday_1hour",
        path="historical-chart/1hour",
        min_tier=Tier.PREMIUM,
        interval_path=True,
        description="1-hour OHLCV.",
    ),
    # --- Catalyst / context -------------------------------------------------
    Endpoint(
        key="stock_news",
        path="news/stock",
        min_tier=Tier.PREMIUM,
        description="Per-symbol news headlines with publish timestamps.",
    ),
    Endpoint(
        key="news_sentiment",
        path="news/stock-latest",
        min_tier=Tier.PREMIUM,
        description="Stock news feed; sentiment join for catalyst tagging.",
    ),
    Endpoint(
        key="press_releases",
        path="news/press-releases",
        min_tier=Tier.PREMIUM,
        description="Official company press releases.",
    ),
    Endpoint(
        key="earnings_calendar",
        path="earnings-calendar",
        min_tier=Tier.PREMIUM,
        symbol_param=None,
        description="Market-wide earnings dates; filtered to symbol locally.",
    ),
    Endpoint(
        key="earnings_symbol",
        path="earnings",
        min_tier=Tier.PREMIUM,
        description="Per-symbol earnings history (actual vs estimate, by announce date).",
    ),
    Endpoint(
        key="income_statement",
        path="income-statement",
        min_tier=Tier.PREMIUM,
        description="Income statements. Carries acceptedDate (the AS-OF date) — use "
        "it, never the fiscal period end, to avoid fundamental lookahead.",
    ),
    Endpoint(
        key="delisted_companies",
        path="delisted-companies",
        min_tier=Tier.PREMIUM,
        symbol_param=None,
        description="Delisted tickers + delist date. Survivorship-bias-free universe.",
    ),
    Endpoint(
        key="analyst_estimates",
        path="analyst-estimates",
        min_tier=Tier.PREMIUM,
        description="Forward analyst estimates.",
    ),
    Endpoint(
        key="price_target",
        path="price-target-summary",
        min_tier=Tier.PREMIUM,
        description="Analyst price-target consensus.",
    ),
    Endpoint(
        key="ratings_historical",
        path="ratings-historical",
        min_tier=Tier.PREMIUM,
        description="Historical company ratings.",
    ),
    Endpoint(
        key="grades",
        path="grades",
        min_tier=Tier.PREMIUM,
        description="Up/down-grade actions with dates. Catalyst tagging.",
    ),
    Endpoint(
        key="insider_trading",
        path="insider-trading/search",
        min_tier=Tier.PREMIUM,
        description="Insider buy/sell transactions.",
    ),
    # --- Ultimate-gated context --------------------------------------------
    Endpoint(
        key="institutional_13f",
        # FMP migrated 13F to the /stable API; the prior 'holdings' path 404s,
        # which made tier auto-detection silently downgrade real Ultimate keys to
        # Premium and block 1min intraday. 'institutional-ownership/latest' is the
        # live Ultimate-gated path (market-wide latest filings; page/limit, no
        # symbol). Used solely as the Ultimate tier probe in client._detect_tier.
        path="institutional-ownership/latest",
        min_tier=Tier.ULTIMATE,
        symbol_param=None,
        description="13F institutional holdings (latest filings). Ultimate tier probe.",
    ),
    Endpoint(
        key="earnings_transcript",
        path="earning-call-transcript",
        min_tier=Tier.ULTIMATE,
        description="Earnings call transcripts.",
    ),
    # --- Macro / reference --------------------------------------------------
    Endpoint(
        key="treasury_rates",
        path="treasury-rates",
        min_tier=Tier.PREMIUM,
        symbol_param=None,
        description="Treasury yield curve. Rate-regime context.",
    ),
    Endpoint(
        key="economic_calendar",
        path="economic-calendar",
        min_tier=Tier.PREMIUM,
        symbol_param=None,
        description="Macro release calendar. Market-wide catalyst context.",
    ),
    Endpoint(
        key="sector_performance",
        path="sector-performance-snapshot",
        min_tier=Tier.PREMIUM,
        symbol_param=None,
        description="Sector performance. Relative-strength context.",
    ),
    Endpoint(
        key="profile",
        path="profile",
        min_tier=Tier.FREE,
        description="Company profile (sector, industry, beta, shares out).",
    ),
    Endpoint(
        key="stock_splits",
        path="splits",
        min_tier=Tier.FREE,
        description="Split history. Adjustment integrity checks.",
    ),
)

ENDPOINTS: Final[dict[str, Endpoint]] = {e.key: e for e in _ENDPOINTS}


def get_endpoint(key: str) -> Endpoint:
    try:
        return ENDPOINTS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown endpoint {key!r}. Known: {', '.join(sorted(ENDPOINTS))}.") from exc


def available_endpoints(tier: Tier) -> dict[str, Endpoint]:
    """All endpoint keys accessible at the given tier."""
    return {k: e for k, e in ENDPOINTS.items() if tier >= e.min_tier}


def gated_endpoints(tier: Tier) -> dict[str, Endpoint]:
    """Endpoints the active tier cannot reach (reported as UNAVAILABLE)."""
    return {k: e for k, e in ENDPOINTS.items() if tier < e.min_tier}
