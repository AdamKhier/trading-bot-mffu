from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Product specs: point value ($ per 1.0 index point), tick size, fee per side.
PRODUCTS: dict[str, dict[str, float]] = {
    "MNQ": {"point_value": 2.0, "tick": 0.25, "fee_per_side": 1.23},
    "MES": {"point_value": 5.0, "tick": 0.25, "fee_per_side": 1.30},
    "NQ": {"point_value": 20.0, "tick": 0.25, "fee_per_side": 2.49},
    "ES": {"point_value": 50.0, "tick": 0.25, "fee_per_side": 2.51},
}

# Trading hours, Eastern Time.
RTH_OPEN_ET = "09:30"
RTH_CLOSE_ET = "16:00"
GLOBEX_OPEN_ET = "18:00"  # prior evening session start

# UTC anchor for time manipulation
UTC = "UTC"
EASTERN = "US/Eastern"


@dataclass
class RiskParams:
    """Position sizing and execution cost assumptions."""

    risk_per_trade: float = 50.0  # USD risked per trade based on stop distance
    max_contracts: int = 3
    slippage_ticks: int = 1


@dataclass
class MFFUEvalParams:
    """MFFU evaluation rulebook parameters.

    drawdown_type:
      - "eod_trailing"     trailing max loss measured at end of day (Pro/Builder style)
      - "intraday_trailing" trailing max loss measured continuously intraday (Rapid style)
    """

    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    drawdown_type: str = "eod_trailing"
    max_loss: float = 2_000.0
    consistency_factor: float | None = 0.5  # best day / total profit must be <= this
    min_trading_days: int = 1
    flat_time_et: str = "15:57"  # open positions flat by this time ET


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    report_dir: Path = field(default_factory=lambda: ROOT / "reports")
    api_key_env: str = "DATABENTO_API_KEY"
    default_start: str = "2020-01-01"
    default_end: str = "2026-09-01"
    train_end: str = "2024-12-31"  # walk-forward split: train <= this, OOS after
    symbols: tuple[str, ...] = ("MNQ", "MES")
    risk: RiskParams = field(default_factory=RiskParams)
    eval: MFFUEvalParams = field(default_factory=MFFUEvalParams)


DEFAULT_CONFIG = Config()


def product_spec(symbol: str) -> dict[str, float]:
    if symbol not in PRODUCTS:
        raise KeyError(f"unknown product {symbol!r}; known: {sorted(PRODUCTS)}")
    return PRODUCTS[symbol]