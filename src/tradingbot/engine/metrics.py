"""Performance metrics computed from trade list + daily P&L series."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    n_days: int = 0
    n_trades: int = 0
    net_pnl: float = 0.0
    fees_paid: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_dd_pct: float = 0.0
    max_dd_usd: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    sharpe: float = 0.0
    avg_trades_day: float = 0.0
    avg_hold_min: float = 0.0
    worst_trade: float = 0.0
    best_trade: float = 0.0
    consistency_ratio: float = 0.0
    daily_pnl: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def compute_metrics(trades: list, daily_pnl: pd.Series) -> Metrics:
    m = Metrics()
    m.n_trades = len(trades)
    m.fees_paid = float(sum(t.fees for t in trades))
    m.net_pnl = float(sum(t.pnl for t in trades))

    win = [t.pnl for t in trades if t.pnl > 0]
    loss = [t.pnl for t in trades if t.pnl <= 0]
    if trades:
        m.win_rate = len(win) / len(trades)
        m.avg_win = float(np.mean(win)) if win else 0.0
        m.avg_loss = float(np.mean(loss)) if loss else 0.0
    if loss:
        m.profit_factor = abs(sum(win)) / abs(sum(loss))
    else:
        m.profit_factor = float("inf") if sum(win) > 0 else 0.0
    m.expectancy = m.net_pnl / len(trades) if trades else 0.0
    if trades:
        m.avg_hold_min = float(np.mean([t.hold_min for t in trades]))
        m.worst_trade = min(t.pnl for t in trades)
        m.best_trade = max(t.pnl for t in trades)

    if len(daily_pnl) == 0:
        return m

    vals = daily_pnl.dropna()
    m.daily_pnl = [float(x) for x in vals.to_list() if abs(x) > 1e-9]
    m.n_days = len(m.daily_pnl)
    eq = vals.cumsum()
    dd = eq - eq.cummax()
    m.max_dd_usd = float(dd.min()) if len(dd) else 0.0
    denom = eq.cummax()
    m.max_dd_pct = float((dd / denom.replace(0, np.nan)).min()) if len(dd) else 0.0
    m.best_day = max(m.daily_pnl, default=0.0)
    m.worst_day = min(m.daily_pnl, default=0.0)
    total_win = sum(x for x in m.daily_pnl if x > 0)
    m.consistency_ratio = (m.best_day / total_win) if total_win > 1e-9 else 0.0
    m.avg_trades_day = m.n_trades / m.n_days if m.n_days else 0.0

    rets = np.asarray(m.daily_pnl, dtype=float)
    if len(rets) > 2 and rets.std() > 1e-12:
        m.sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    return m