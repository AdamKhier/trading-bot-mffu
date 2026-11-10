"""Strategy C: Midday Liquidity Regime.

Thesis: around 11:00-13:30 ET order book structure shifts materially - trade count
drops, per-print size survives, realized range collapses. Uniform-clock bots misfire
here. Two complementary edges:
  1. The tight "regime range" pin absorbs orders; once flow re-expands, the first
     wide bar closing beyond the pin range is a stop-run continuation (follow).
  2. While the pin persists, thin drift establishes and drifts away from the regime
     baseline (trend-following a low-liquidity auction).

RTH only, max 1 trade/day; exits forced by 13:35 ET.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.strategies.base import Strategy


class MiddayLiquidity(Strategy):
    name = "midday_liquidity"
    display_name = "Midday Liquidity Regime"
    max_trades_per_day = 1
    grid_keys = ["thin_pct", "breakout_mult"]
    default_params = {
        "thin_pct": [0.45, 0.55, 0.65],
        "breakout_mult": [1.8, 2.5],
        "drift_bars": 3,
        "rr": 1.2,
        "start_et": "11:00",
        "end_et": "13:35",
    }

    def build_signals(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sym = list(bars)[0]
        full = bars[sym]
        r = full[full["in_rth"]].copy()

        hl = r["high"] - r["low"]
        med_vol = _group_expanding_median(r["volume"], r["trade_date"])
        med_range = _group_expanding_median(hl, r["trade_date"])

        thin = (r["volume"] <= self.params["thin_pct"] * med_vol) & (hl <= 0.9 * med_range)
        regime = thin.rolling(12, min_periods=8).sum() >= 8

        group_id = (regime & ~regime.shift(fill_value=False)).cumsum().where(regime)
        range_low = r["low"].groupby(group_id).cummin()
        range_high = r["high"].groupby(group_id).cummax()
        baseline = r["close"].groupby(group_id).transform("first")

        above = (r["close"] > baseline) & regime
        below = (r["close"] < baseline) & regime
        above3 = above & above.shift(fill_value=False) & above.shift(2, fill_value=False)
        below3 = below & below.shift(fill_value=False) & below.shift(2, fill_value=False)

        wide = hl > self.params["breakout_mult"] * med_range
        break_hi = wide & (r["close"] > range_high)
        break_lo = wide & (r["close"] < range_low)

        long_entry = (above3 & ~above3.shift(fill_value=False)) | break_hi
        short_entry = (below3 & ~below3.shift(fill_value=False)) | break_lo

        rr = self.params["rr"]
        comp = r["close"] - range_low  # distance to regime low
        comp2 = range_high - r["close"]  # distance to regime high
        out = pd.DataFrame(index=r.index)
        out["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0)).astype(int)
        stop = np.where(
            (out["signal"] == 1).to_numpy(),
            (comp / 0.25).clip(lower=1),
            (comp2 / 0.25).clip(lower=1),
        )
        out["stop_dist_ticks"] = stop
        out["target_dist_ticks"] = np.maximum(stop * rr, 1)

        et = r["time_et"].dt.strftime("%H:%M")
        out["valid"] = (et >= self.params["start_et"]) & (et <= self.params["end_et"]) & regime.fillna(False).astype(bool)
        out["hard_exit"] = et >= self.params["end_et"]

        return _finish(out, full.index)


def _group_expanding_median(s: pd.Series, trade_date: pd.Series) -> pd.Series:
    m = s.groupby(trade_date).expanding().median().reset_index(level=0, drop=True)
    return m.reindex(s.index).fillna(s.median())


def _finish(out: pd.DataFrame, full_index: pd.Index) -> pd.DataFrame:
    out = out.reindex(full_index)
    out["signal"] = out["signal"].fillna(0).astype(int)
    out["stop_dist_ticks"] = out["stop_dist_ticks"].fillna(1).clip(lower=1)
    out["target_dist_ticks"] = out["target_dist_ticks"].fillna(1).clip(lower=1)
    out["valid"] = out["valid"].fillna(False).astype(bool)
    out["hard_exit"] = out["hard_exit"].fillna(False).astype(bool)
    # NaN regime-derived columns outside RTH: fill
    return out