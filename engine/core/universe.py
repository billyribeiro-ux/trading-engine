"""
Universe metadata: instrument type + inverse/leverage relationships.

Per the project's scanner plan, the universe must flag each member's type so
features and handling can be type-aware. The decisive rule for inverse/leveraged
ETFs (SQQQ, TQQQ, SOXL, …): they reset leverage DAILY, so over multi-day/-week
holds they do NOT equal ±Nx the underlying's cumulative move (they compound and
bleed). Therefore labels are ALWAYS computed on the ETF's OWN price path — which
the pipeline already does (build_*_frame fetches the instrument's own bars). This
metadata is for FEATURES (e.g. relative strength vs the thing it inverts) and
type-aware handling (liquidity, reverse-split frequency), NOT for substituting
"short the underlying" math.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstrumentType(str, Enum):
    SINGLE_STOCK = "single_stock"
    INDEX_ETF = "index_etf"
    SECTOR_ETF = "sector_etf"
    FACTOR_ETF = "factor_etf"
    INVERSE_ETF = "inverse_etf"
    LEVERAGED_ETF = "leveraged_etf"


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    instrument_type: InstrumentType = InstrumentType.SINGLE_STOCK
    # The instrument this one tracks/inverts (e.g. SQQQ -> QQQ); None for plain.
    inverse_of: str | None = None
    # Signed daily leverage: SQQQ -3, TQQQ +3, SH -1, plain 1.
    leverage_factor: float = 1.0
    # Survivorship: delisted names MUST be scannable. Their absence from a screen
    # is the classic survivorship bias — the losers that died inflate a winners-
    # only backtest. They still carry data up to their delist date (labels stop
    # there naturally), so they belong in the universe, flagged.
    delisted: bool = False
    delisted_date: str | None = None

    @property
    def is_derived(self) -> bool:
        """A leverage/inverse wrapper whose fundamentals belong to the basket,
        not the ticker — and whose outcome must be read off its OWN bars."""
        return self.inverse_of is not None or abs(self.leverage_factor) != 1.0


# Seed registry of common leveraged/inverse products + a few benchmarks. Extend
# as the scanned universe grows; unknown symbols default to a plain single stock.
_REGISTRY: dict[str, UniverseMember] = {
    "SPY": UniverseMember("SPY", InstrumentType.INDEX_ETF),
    "QQQ": UniverseMember("QQQ", InstrumentType.INDEX_ETF),
    "SOXX": UniverseMember("SOXX", InstrumentType.SECTOR_ETF),
    "SH": UniverseMember("SH", InstrumentType.INVERSE_ETF, "SPY", -1.0),
    "SDS": UniverseMember("SDS", InstrumentType.INVERSE_ETF, "SPY", -2.0),
    "SPXU": UniverseMember("SPXU", InstrumentType.INVERSE_ETF, "SPY", -3.0),
    "UPRO": UniverseMember("UPRO", InstrumentType.LEVERAGED_ETF, "SPY", 3.0),
    "SQQQ": UniverseMember("SQQQ", InstrumentType.INVERSE_ETF, "QQQ", -3.0),
    "TQQQ": UniverseMember("TQQQ", InstrumentType.LEVERAGED_ETF, "QQQ", 3.0),
    "SOXL": UniverseMember("SOXL", InstrumentType.LEVERAGED_ETF, "SOXX", 3.0),
    "SOXS": UniverseMember("SOXS", InstrumentType.INVERSE_ETF, "SOXX", -3.0),
}


def classify(symbol: str) -> UniverseMember:
    """Universe metadata for a symbol (defaults to a plain single stock)."""
    return _REGISTRY.get(symbol.strip().upper(), UniverseMember(symbol.strip().upper()))


def fetch_delisted(client, limit: int = 100) -> list[UniverseMember]:
    """Delisted tickers (with delist date), flagged. Include these in a screen so
    the backtest isn't survivorship-biased toward names that are still alive."""
    out: list[UniverseMember] = []
    try:
        df = client.fetch("delisted_companies", params={"page": 0, "limit": limit})
    except Exception:
        return out
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "")).strip().upper()
        if not sym:
            continue
        base = classify(sym)
        out.append(
            UniverseMember(
                symbol=sym,
                instrument_type=base.instrument_type,
                inverse_of=base.inverse_of,
                leverage_factor=base.leverage_factor,
                delisted=True,
                delisted_date=(str(r["delistedDate"]) if r.get("delistedDate") else None),
            )
        )
    return out


def build_universe(
    symbols: list[str], client=None, include_delisted: bool = False, delisted_limit: int = 100
) -> list[UniverseMember]:
    """Assemble a scannable universe. With include_delisted (and a client), append
    recently-delisted names so the screen is survivorship-bias-free. NOTE: this is
    the inclusion hook; full POINT-IN-TIME index reconstitution (membership as of a
    historical date) is the deeper data layer and is not done here — but scanning
    only survivors is the bias to avoid, and this lets you not do that."""
    members = [classify(s) for s in symbols]
    seen = {m.symbol for m in members}
    if include_delisted and client is not None:
        for m in fetch_delisted(client, limit=delisted_limit):
            if m.symbol not in seen:
                members.append(m)
                seen.add(m.symbol)
    return members
