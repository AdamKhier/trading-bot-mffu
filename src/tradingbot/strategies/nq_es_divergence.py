"""Strategy B: NQ/ES Statistical Divergence.

Thesis: the Nasdaq-100 and S&P 500 index complexes share near-common drift but
temporarily deviate from the stale rolling beta as different program flows hit each
complex. Index-arbitrage flow pulls the residual back. Retail reads single-instrument
charts; almost nobody encodes the *joint* structure.

Rule (RTH only): exp-weighted beta of log(MNQ) on log(MES); z-score the residual with
the same decay. When MNQ is statistically rich vs ES (z >= +entry_z) fade it short;
mirror when cheap. Exit on reversion (|z| <= z_exit) or protective stop
(z_stop_delta residual-stdevs beyond entry). Signals masked around CME quarterly rolls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.config import PRODUCTS
from tradingbot.strategies.base import Strategy, ema_beta

ROLL_MONTHS = {3, 6, 9, 12}


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=int(days_to_friday + 14))


def _is_roll_date(d: object) -> bool:
    try:
        ts = pd.Timestamp(d)
    except Exception:
        return False
    if ts.month not in ROLL_MONTHS:
        return False
    return ts.date() == _third_friday(ts.year, ts.month).date()


class NQESDivergence(Strategy):
    name = "nq_es_divergence"
    display_name = "NQ/ES Statistical Divergence"
    grid_keys = ["half_life", "z_entry"]
    default_params = {
        "half_life": [30, 60, 120],
        "z_entry": [1.75, 2.25],
        "z_exit": 0.25,
        "z_stop_delta": 1.0,
        "rr": 1.2,
        "start_et": "09:45",
        "end_et": "15:15",
        "leg": "MNQ",
    }

    def requires(self) -> list[str]:
        return ["MNQ", "MES"]

    def build_signals(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        leg = self.params["leg"]
        other = "MES" if leg == "MNQ" else "MNQ"
        A, B = bars[leg], bars[other]
        tick = PRODUCTS[leg]["tick"]

        idx = A.index.intersection(B.index)
        common = A.reindex(idx).join(B[["close"]].rename(columns={"close": "bclose"}), how="inner")
        rth = common[common["in_rth"]].copy()

        lm = np.log(rth["close"])
        le = np.log(rth["bclose"])
        beta = ema_beta(lm.diff(), le.diff(), self.params["half_life"]).shift()
        alpha = 1.0 - 2 ** (-1.0 / self.params["half_life"])
        resid = lm - beta * le
        mu = resid.ewm(alpha=alpha, adjust=False).mean()
        sd = resid.ewm(alpha=alpha, adjust=False).std()
        z = (resid - mu) / (sd + 1e-12)

        sd_pts = sd * rth["close"]
        stop_ticks = np.ceil((self.params["z_stop_delta"] * sd_pts) / tick)

        ze = self.params["z_entry"]
        short = (z >= ze) & (z.shift() < ze)
        long = (z <= -ze) & (z.shift() > -ze)
        exitz = z.abs() <= self.params["z_exit"]
        stop_out = ((z >= ze + self.params["z_stop_delta"]) | (z <= -ze - self.params["z_stop_delta"]))

        sig = np.zeros(len(rth), dtype=int)
        pos = 0
        for i in range(len(rth)):
            if pos == 0:
                if short.iloc[i]:
                    pos = -1
                elif long.iloc[i]:
                    pos = 1
            else:
                if exitz.iloc[i] or stop_out.iloc[i]:
                    pos = 0
            sig[i] = pos

        out = pd.DataFrame(index=rth.index)
        out["signal"] = sig
        out["stop_dist_ticks"] = stop_ticks.clip(lower=1)
        out["target_dist_ticks"] = (stop_ticks * self.params["rr"]).clip(lower=1)

        et = rth["time_et"].dt.strftime("%H:%M")
        roll_mask = rth["trade_date"].map(_is_roll_date).astype(bool)
        out["valid"] = (
            (et >= self.params["start_et"])
            & (et <= self.params["end_et"])
            & ~roll_mask
            & (np.arange(len(rth)) > 240)
        )
        out["hard_exit"] = et >= self.params["end_et"]

        return _finish(out, A.index)


def _finish(out: pd.DataFrame, full_index: pd.Index) -> pd.DataFrame:
    out = out.reindex(full_index)
    out["signal"] = out["signal"].fillna(0).astype(int)
    out["stop_dist_ticks"] = out["stop_dist_ticks"].fillna(1).clip(lower=1)
    out["target_dist_ticks"] = out["target_dist_ticks"].fillna(1).clip(lower=1)
    out["valid"] = out["valid"].fillna(False).astype(bool)
    out["hard_exit"] = out["hard_exit"].fillna(False).astype(bool)
    return out