"""Strategy registry + module exports."""
from __future__ import annotations

from tradingbot.strategies.auction_exhaustion import AuctionExhaustion
from tradingbot.strategies.base import Strategy
from tradingbot.strategies.midday_liquidity import MiddayLiquidity
from tradingbot.strategies.nq_es_divergence import NQESDivergence
from tradingbot.strategies.overnight_handoff import OvernightHandoff

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        AuctionExhaustion,
        NQESDivergence,
        MiddayLiquidity,
        OvernightHandoff,
    )
}


def get_strategy(name: str, params: dict | None = None) -> Strategy:
    cls = STRATEGIES[name]
    return cls(params or {})