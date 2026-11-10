"""Strategy A: Auction Exhaustion Index.

Thesis: prices that travel far on *thin flow density* (distance per unit volume) are
unbalanced auctions. Retail trades the distance, ignoring flow density; sustained
turnover with weak flow reverts. We fade extreme, flow-poor stretches toward the
anchor mean.

Rule (RTH only): z-score the N-min move / N-min flow ratio on a long lookback. When
the up-stretch z-score exceeds the threshold the push is flow-poor -> short; mirror
for longs. Requires a minimum raw move (ticks) so tiny noise bars never trigger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.strategies.base import Strategy, zscore_ts


class AuctionExhaustion(Strategy):
    name = "auction_exhaustion"
    display_name = "Auction Exhaustion Index"
    grid_keys = ["window", "z_thresh"]
    default_params = {
        "window": [15, 30, 60],
        "z_thresh": [1.6, 2.0, 2.4],
        "lookback": 480,      # long-term z-score reference (minutes)
        "stop_ticks": 14,
        "rr": 1.25,
        "min_move_ticks": 12, # distance traveled must exceed this (ticks)
        "start_et": "09:45",
        "end_et": "15:15",
    }

    def build_signals(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sym = list(bars)[0]
        full = bars[sym]
        r = full[full["in_rth"]].copy()
        tick = 0.25

        d = r["close"].diff()
        up, down = d.clip(lower=0), (-d).clip(lower=0)
        N = self.params["window"]
        flow = r["volume"].rolling(N, min_periods=max(2, N // 2)).sum().replace(0, 1.0)
        upflow = up.rolling(N, min_periods=max(2, N // 2)).sum() / flow
        downflow = down.rolling(N, min_periods=max(2, N // 2)).sum() / flow

        z_up = zscore_ts(upflow, self.params["lookback"])
        z_down = zscore_ts(downflow, self.params["lookback"])
        th = self.params["z_thresh"]

        move_up = r["close"] - r["close"].shift(N)  # points pushed up over window
        move_dn = -move_up
        min_move = self.params["min_move_ticks"] * tick

        short = (z_up > th) & (z_up.shift() <= th) & (move_up >= min_move)
        long = (z_down > th) & (z_down.shift() <= th) & (move_dn >= min_move)

        out = pd.DataFrame(index=r.index)
        out["signal"] = np.where(short, -1, np.where(long, 1, 0)).astype(int)
        out["stop_dist_ticks"] = np.full(len(r), float(self.params["stop_ticks"]))
        out["target_dist_ticks"] = round(self.params["stop_ticks"] * self.params["rr"])

        et = r["time_et"].dt.strftime("%H:%M")
        out["valid"] = (
            (et >= self.params["start_et"])
            & (et <= self.params["end_et"])
            & (np.arange(len(r)) > self.params["lookback"] * 2)
        )

        out["hard_exit"] = et >= self.params["end_et"]

        out = out.reindex(full.index)
        out["signal"] = out["signal"].fillna(0).astype(int)
        out["stop_dist_ticks"] = out["stop_dist_ticks"].fillna(1).clip(lower=1)
        out["target_dist_ticks"] = out["target_dist_ticks"].fillna(1).clip(lower=1)
        out["valid"] = out["valid"].fillna(False).astype(bool)
        out["hard_exit"] = out["hard_exit"].fillna(False).astype(bool)
        return out