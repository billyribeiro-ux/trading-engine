"""
Execution cost and fill modeling.

A backtest that fills at the exact signal price with no friction reports an edge
that does not exist live. This module applies the costs that decide whether a
reversal edge survives contact with the market:

    * Commission        -- per-share or per-trade, both supported.
    * Spread            -- you buy at the ask, sell at the bid. Half-spread is
                           paid on entry and again on exit.
    * Slippage          -- market/stop orders fill worse than the trigger price,
                           scaled by volatility (ATR) and optionally by how thin
                           the bar's volume is.
    * Stop-vs-target intrabar resolution -- when a single bar's range spans both
                           the stop and a target, we already assume stop-first
                           (pessimistic). The cost model additionally fills the
                           stop with slippage *beyond* the stop price, because a
                           stop becomes a market order when touched.

The model is deliberately conservative. The institutional failure mode is an
optimistic backtest; we bias the other way so a surviving edge is a real one.

All parameters are explicit and documented. Defaults approximate a liquid US
large-cap on a retail-plus routing arrangement; tune per instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class CostModel:
    """
    Execution cost parameters.

    commission_per_share : $ per share (e.g. 0.005). Set 0 if using per-trade.
    commission_per_trade : flat $ per fill (e.g. 1.0). Applied on entry and exit.
    min_commission       : floor per fill when using per-share.
    half_spread_bps      : half the bid-ask spread in basis points of price.
                           Paid on entry and on exit (you cross the spread both
                           times). 1 bp = 0.01%.
    slippage_atr_frac    : slippage as a fraction of ATR, applied to market/stop
                           fills. 0.05 means a stop fills 5% of an ATR worse than
                           its trigger. This is the dominant cost for stops.
    entry_is_marketable  : if True, entry pays half-spread + slippage (you take
                           liquidity on the reversal confirmation). If False,
                           entry is a passive limit at the signal price (rare for
                           a momentum-confirmation entry; default True).
    """

    commission_per_share: float = 0.005
    commission_per_trade: float = 0.0
    min_commission: float = 1.0
    half_spread_bps: float = 1.0
    slippage_atr_frac: float = 0.05
    entry_is_marketable: bool = True

    def _commission(self, shares: float, notional: float) -> float:
        if self.commission_per_trade > 0:
            return self.commission_per_trade
        return max(self.min_commission, abs(shares) * self.commission_per_share)

    def entry_fill(self, side: Side, signal_price: float, atr: float) -> float:
        """
        Effective entry price after spread + slippage (worse than signal_price).

        Long pays UP (toward ask + slippage); short sells DOWN (toward bid -
        slippage). Returns the price actually transacted.
        """
        half_spread = signal_price * self.half_spread_bps / 1e4
        slip = (atr * self.slippage_atr_frac) if self.entry_is_marketable else 0.0
        if side is Side.LONG:
            return signal_price + half_spread + slip
        return signal_price - half_spread - slip

    def stop_fill(self, side: Side, stop_price: float, atr: float) -> float:
        """
        Effective stop fill -- worse than the stop trigger, because a touched
        stop becomes a market order in fast tape. Long stops fill BELOW the stop;
        short stops fill ABOVE it.
        """
        half_spread = stop_price * self.half_spread_bps / 1e4
        slip = atr * self.slippage_atr_frac
        if side is Side.LONG:
            return stop_price - half_spread - slip
        return stop_price + half_spread + slip

    def target_fill(self, side: Side, target_price: float) -> float:
        """
        Target fills assume a resting limit order at the target -- you provide
        liquidity, so you do NOT pay slippage, but you still concede half-spread
        because the limit sits at the target and the marketable counterparty
        crosses to you only at the level. Conservative: include half-spread.
        Long sells at target - half_spread; short buys at target + half_spread.
        """
        half_spread = target_price * self.half_spread_bps / 1e4
        if side is Side.LONG:
            return target_price - half_spread
        return target_price + half_spread

    def round_trip_pnl(
        self,
        side: Side,
        entry_signal: float,
        exit_price: float,
        atr: float,
        shares: float = 100.0,
        exit_is_stop: bool = False,
    ) -> dict[str, float]:
        """
        Full round-trip P&L in dollars and in R, net of all costs.

        Returns a dict with gross_pnl, costs, net_pnl, net_pnl_per_share, and the
        effective entry/exit prices, so the backtester can report both gross and
        net edge side by side -- the gap between them is the friction the live
        scanner must clear.
        """
        eff_entry = self.entry_fill(side, entry_signal, atr)
        if exit_is_stop:
            eff_exit = self.stop_fill(side, exit_price, atr)
        else:
            eff_exit = self.target_fill(side, exit_price)

        if side is Side.LONG:
            gross_per_share = exit_price - entry_signal
            net_per_share = eff_exit - eff_entry
        else:
            gross_per_share = entry_signal - exit_price
            net_per_share = eff_entry - eff_exit

        notional = abs(eff_entry * shares)
        commission = (
            self._commission(shares, notional)  # entry
            + self._commission(shares, abs(eff_exit * shares))  # exit
        )
        gross_pnl = gross_per_share * shares
        net_pnl = net_per_share * shares - commission

        return {
            "eff_entry": eff_entry,
            "eff_exit": eff_exit,
            "gross_pnl": gross_pnl,
            "commission": commission,
            "spread_slippage_cost": (gross_pnl - (net_per_share * shares)),
            "net_pnl": net_pnl,
            "net_pnl_per_share": net_pnl / shares if shares else 0.0,
        }


# A frictionless model, for explicitly computing the gross/net gap.
FRICTIONLESS = CostModel(
    commission_per_share=0.0,
    commission_per_trade=0.0,
    min_commission=0.0,
    half_spread_bps=0.0,
    slippage_atr_frac=0.0,
    entry_is_marketable=False,
)
