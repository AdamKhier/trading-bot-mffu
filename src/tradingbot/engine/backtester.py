"""Bar-by-bar simulator for intraday futures strategies."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradingbot.config import RiskParams, product_spec


@dataclass
class Trade:
    symbol: str
    side: int  # +1 long, -1 short
    contracts: int
    entry: float
    exit_price: float
    enter_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    exit_reason: str
    pnl: float
    fees: float
    hold_min: float


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    bar_equity: pd.Series = field(default_factory=pd.Series)  # per-UTC-bar cumulative equity
    daily_pnl: pd.Series = field(default_factory=pd.Series)  # indexed by trade date

    @property
    def net_pnl(self) -> float:
        return float(sum(t.pnl for t in self.trades))


def _flat_minutes(time_et: str) -> int:
    h, m = time_et.split(":")
    return int(h) * 60 + int(m)


def run_backtest(
    bars: pd.DataFrame,
    sig: pd.DataFrame,
    symbol: str,
    risk: RiskParams | None = None,
    flat_time_et: str = "15:57",
    max_trades_per_day: int = 2,
    cooldown_bars: int = 3,
) -> BacktestResult:
    """Simulate a strategy over bars using the signal frame.

    2022 signals frame columns (index aligned to bars):
      signal          int   target position {-1, 0, +1} at bar close
      stop_dist_ticks float distance from entry to stop, in ticks (>=1)
      target_dist_ticks float distance from entry to target, in ticks
      valid           bool  (optional) entries allowed at this bar
      hard_exit       bool  (optional) exit at this bar's close regardless of stop/target

    Entries fill at the NEXT bar's open (+/- slippage); stops/targets checked against
    bar high/low; positions forced flat by flat_time_et.
    """
    risk = risk or RiskParams()
    spec = product_spec(symbol)
    pv = spec["point_value"]
    tick = spec["tick"]
    fee_side = spec["fee_per_side"]
    slip = risk.slippage_ticks * tick
    flat_min = _flat_minutes(flat_time_et)

    sig = sig.reindex(bars.index)
    valid = np.asarray(sig["valid"].fillna(True).astype(bool)) if "valid" in sig else np.ones(len(sig), dtype=bool)
    hard_exit = np.asarray(sig["hard_exit"].fillna(False).astype(bool)) if "hard_exit" in sig else np.zeros(len(sig), dtype=bool)
    signal = np.asarray(sig["signal"].to_numpy(dtype=float))
    stop_t = np.asarray(sig["stop_dist_ticks"].to_numpy(dtype=float))
    tgt_t = np.asarray(sig["target_dist_ticks"].to_numpy(dtype=float))
    signal = np.nan_to_num(signal, nan=0.0)
    stop_t = np.nan_to_num(stop_t, nan=1.0)
    tgt_t = np.nan_to_num(tgt_t, nan=1.0)

    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    mm_of_day = (bars["time_et"].dt.hour * 60 + bars["time_et"].dt.minute).to_numpy(dtype=int)
    dates = bars["trade_date"].to_numpy()

    n = len(bars)
    trades: list[Trade] = []
    pnl = np.zeros(n, dtype=float)  # realized cumulative
    equity = np.zeros(n, dtype=float)

    pos = 0
    contracts = 0
    entry = 0.0
    stop = 0.0
    target = 0.0
    entry_ts: pd.Timestamp | None = None
    entry_i = 0
    day_trades = 0
    prev_date: object = None
    since_exit = 10 ** 9

    realized = 0.0

    for i in range(n):
        d = dates[i]
        if prev_date != d:
            day_trades = 0
            prev_date = d
        since_exit += 1

        # ---- exits ----
        if pos != 0:
            reason = None
            exit_px = 0.0
            if pos == 1:
                if l[i] <= stop:
                    reason = "stop"
                    exit_px = stop
                elif h[i] >= target:
                    reason = "target"
                    exit_px = target
            else:
                if h[i] >= stop:
                    reason = "stop"
                    exit_px = stop
                elif l[i] <= target:
                    reason = "target"
                    exit_px = target
            if reason is None and (hard_exit[i] or mm_of_day[i] >= flat_min):
                reason = "time"
                exit_px = c[i]
            if reason is not None:
                gross = (exit_px - entry) * pv * contracts * pos
                fees = fee_side * contracts * 2.0
                nt = gross - fees
                realized += nt
                trades.append(
                    Trade(
                        symbol=symbol,
                        side=pos,
                        contracts=contracts,
                        entry=entry,
                        exit_price=exit_px,
                        enter_ts=entry_ts or bars.index[i],
                        exit_ts=bars.index[i],
                        exit_reason=reason,
                        pnl=nt,
                        fees=fees,
                        hold_min=int((bars.index[i] - (entry_ts or bars.index[i])).total_seconds() / 60),
                    )
                )
                pos = 0
                day_trades += 1
                since_exit = 0

        # ---- entries ----
        if pos == 0 and signal[i] != 0 and valid[i] and day_trades < max_trades_per_day and since_exit >= cooldown_bars:
            if mm_of_day[i] < flat_min and i + 1 < n:
                side = 1 if signal[i] > 0 else -1
                entry_px = o[i + 1] + side * slip
                sd = max(float(stop_t[i]), 1.0) * tick
                td = max(float(tgt_t[i]), 1.0) * tick
                stop_v = sd * pv
                if stop_v <= 0:
                    stop_v = 1.0
                k = int(min(risk.max_contracts, max(1, int(np.ceil(risk.risk_per_trade / stop_v)))))
                pos = side
                contracts = k
                entry = entry_px
                stop = entry - side * sd
                target = entry + side * td
                entry_ts = bars.index[i + 1]
                entry_i = i + 1

        realized_now = realized
        if pos != 0:
            unrealized = (c[i] - entry) * pv * contracts * pos
        else:
            unrealized = 0.0
        equity[i] = realized_now + unrealized
        pnl[i] = realized_now

    # ---- daily aggregation by ET trade date ----
    eq = pd.Series(equity, index=bars.index, name="equity")
    day_of_bar = pd.Series(bars["trade_date"].to_numpy(), index=bars.index)
    daily = eq.groupby(day_of_bar).last()
    daily_pnl = daily.diff()
    if len(daily_pnl):
        daily_pnl.iloc[0] = daily.iloc[0]
    daily_pnl = daily_pnl.rename("pnl")

    return BacktestResult(symbol=symbol, trades=trades, bar_equity=eq, daily_pnl=daily_pnl)