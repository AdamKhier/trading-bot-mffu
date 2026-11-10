"""Strategy D: Settlement/Overnight Handoff.

Thesis: the overnight (Globex) auction is the densest, most mechanical flow period.
Its *geometry* - VWAP position, compression vs history, symmetry - carries into the
opening handoff, but retail reads the printed evening bars, not the structure.

Two legs, evaluated at 09:45 ET (after the first 15 RTH minutes):
  FADE - compressed overnight (range ratio < comp) and the open has drifted far from
         overnight VWAP (>= dev_ticks): fade the drift back toward the VWAP.
  FOLLOW - expanded overnight (range ratio >= exp_thresh) and the open has broken the
         overnight range: follow the range expansion with momentum.

Positions force-exited by 10:15 ET. One signal/day max.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.strategies.base import Strategy


class OvernightHandoff(Strategy):
    name = "overnight_handoff"
    display_name = "Settlement / Overnight Handoff"
    max_trades_per_day = 1
    default_params = {
        "comp": 0.8,
        "dev_ticks": 12,
        "exp_thresh": 1.4,
        "rr": 1.2,
        "signal_et": "09:45",
        "exit_et": "10:15",
        "legs": ["fade", "follow"],
    }

    @classmethod
    def expand_grid(cls) -> list[dict]:
        combos = []
        for comp in (0.7, 0.85):
            for dev in (8, 16):
                combos.append({"comp": comp, "dev_ticks": dev, "legs": ["fade"]})
        for exp in (1.2, 1.5):
            combos.append({"exp_thresh": exp, "legs": ["follow"]})
        return combos

    def build_signals(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        sym = list(bars)[0]
        full = bars[sym]
        tick = 0.25
        legs = self.params["legs"]

        # Per-trade-date overnight attributes from the pre-09:30 segment
        pre = full[full["time_et"].dt.strftime("%H:%M") < "09:30"].copy()
        if len(pre) < 60:
            return _blank(full.index)

        seg = pre[["close", "high", "low", "volume", "trade_date"]].copy()
        seg["pv"] = seg["close"] * seg["volume"]
        daily = seg.groupby("trade_date").agg(
            vwap=("pv", "sum"),
            vol=("volume", "sum"),
            o_hi=("high", "max"),
            o_lo=("low", "min"),
            o_first=("close", "first"),
            o_last=("close", "last"),
        )
        daily["vwap"] /= daily["vol"].replace(0, np.nan)
        daily["o_range"] = daily["o_hi"] - daily["o_lo"]
        # trailing 20-comparable-days median of PRIOR overnight ranges (no lookahead)
        daily["range_med"] = daily["o_range"].shift().rolling(20, min_periods=10).median()
        daily["ratio"] = daily["o_range"] / daily["range_med"]

        # Day open price (first RTH bar open per trade_date)
        rth = full[full["in_rth"]]
        day_open = rth.groupby("trade_date")["open"].first()

        # Signal bar = the first minute of the window (signal_et)
        et = full["time_et"].dt.strftime("%H:%M")
        sig_mask = et == self.params["signal_et"]
        signal_ts = full.index[sig_mask]

        out = pd.DataFrame(index=full.index)
        out["signal"] = 0
        out["stop_dist_ticks"] = 1.0
        out["target_dist_ticks"] = 1.0
        out["valid"] = False
        out["hard_exit"] = et >= self.params["exit_et"]

        if len(signal_ts) == 0:
            return out

        # join daily attrs to signal bars
        sig_bars = full.loc[signal_ts].copy()
        sig_bars["vwap"] = sig_bars["trade_date"].map(daily["vwap"])
        sig_bars["o_range"] = sig_bars["trade_date"].map(daily["o_range"])
        sig_bars["o_hi"] = sig_bars["trade_date"].map(daily["o_hi"])
        sig_bars["o_lo"] = sig_bars["trade_date"].map(daily["o_lo"])
        sig_bars["ratio"] = sig_bars["trade_date"].map(daily["ratio"])
        sig_bars["day_open"] = sig_bars["trade_date"].map(day_open)

        sig_bars = sig_bars.dropna(subset=["vwap", "ratio", "day_open"]).copy()

        for ts, row in sig_bars.iterrows():
            dev_ticks = (row["close"] - row["vwap"]) / tick
            rr = self.params["rr"]

            if "fade" in legs and row["ratio"] < self.params["comp"] and abs(dev_ticks) >= self.params["dev_ticks"]:
                side = 1 if dev_ticks < 0 else -1
                out.loc[ts, "signal"] = side
                out.loc[ts, "stop_dist_ticks"] = max((row["o_range"] * 0.5) / tick, 1.0)
                out.loc[ts, "target_dist_ticks"] = max(abs(dev_ticks), 1.0)
                out.loc[ts, "valid"] = True
            elif "follow" in legs and row["ratio"] >= self.params["exp_thresh"]:
                if row["day_open"] > row["o_hi"]:
                    out.loc[ts, "signal"] = 1
                    stop_t = max(abs(row["day_open"] - row["vwap"]) / tick, 2.0)
                elif row["day_open"] < row["o_lo"]:
                    out.loc[ts, "signal"] = -1
                    stop_t = max(abs(row["day_open"] - row["vwap"]) / tick, 2.0)
                else:
                    stop_t = 0.0
                if stop_t > 0:
                    out.loc[ts, "stop_dist_ticks"] = stop_t
                    out.loc[ts, "target_dist_ticks"] = max(stop_t * rr, 1.0)
                    out.loc[ts, "valid"] = True

            # hard force exit later in the day handled by hard_exit col

        return out


def _blank(index: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["signal"] = 0
    df["stop_dist_ticks"] = 1.0
    df["target_dist_ticks"] = 1.0
    df["valid"] = False
    df["hard_exit"] = False
    return df