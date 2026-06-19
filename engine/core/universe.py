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
