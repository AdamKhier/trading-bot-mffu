"""Synthetic 1-min futures data for pipeline smoke tests (NOT for edge discovery)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.data.records import normalize_bars
from tradingbot.data.provider import DataProvider

RNG_SEED = 7


def _intraday_vol_tmpl(x: np.ndarray) -> np.ndarray:
    """Volume/volatility template across the RTH session (open/close heavy, lunch lite)."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x < 0.2,
        1.8 - 4.5 * x,
        np.where(x < 0.45, 1.04 - 1.6 * x, np.where(x < 0.85, 1.18 - 1.35 * (x - 0.45), 1.15 + 2.6 * (x - 0.85))),
    )


class SyntheticProvider(DataProvider):
    """Generates plausible continuous 1-min bars for MNQ/MES.

    Structure only: an OU-like price path with a shaped intraday volume curve, so the
    pipeline (data -> signals -> backtest -> eval sim) can be smoke-tested end-to-end
    without any API key.
    """

    _seed = {"MNQ": 12_000.0, "MES": 3_000.0, "NQ": 12_000.0, "ES": 3_000.0}
    _daily_vol = {"MNQ": 0.0045, "MES": 0.0040, "NQ": 0.0045, "ES": 0.0040}
    _base_vol = {"MNQ": 380.0, "MES": 520.0, "NQ": 260.0, "ES": 380.0}

    def __init__(self, rng_seed: int = RNG_SEED):
        self.rng = np.random.default_rng(rng_seed)

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        idx = pd.date_range(s, e, freq="min", tz="UTC")
        et = idx.tz_convert("US/Eastern")

        hhmm = et.strftime("%H:%M")
        rth = (hhmm >= "09:30") & (hhmm < "16:00") & (et.dayofweek < 5)

        # Session minute-of-RTH index for the volume template.
        rth_min = np.zeros(len(idx), dtype=int)
        counter = 0
        prev_date = None
        for k, (flag, d) in enumerate(zip(rth, et.normalize())):
            if flag:
                if d != prev_date:
                    counter = 0
                rth_min[k] = counter
                counter += 1
                prev_date = d
        rth_frac = rth_min / 390.0

        daily_vol = self._daily_vol[symbol]
        base_vol = self._base_vol[symbol]

        rnd = self.rng.normal(0, 1, len(idx))
        # Daily anchored random walk: overnight portions get lower vol.
        per_min_vol = np.where(rth, daily_vol * _intraday_vol_tmpl(rth_frac) * 0.85, daily_vol * 0.35)
        log_price = np.log(self._seed[symbol]) + np.cumsum(per_min_vol * rnd)

        o = np.empty(len(idx))
        h = np.empty(len(idx))
        l = np.empty(len(idx))
        c = np.exp(log_price)
        c_roll = np.concatenate([[c[0]], c[:-1]])
        body = (c - c_roll) * 0.5
        openp = c - body
        high = np.maximum(openp, c) + np.abs(body)
        low = np.minimum(openp, c) - np.abs(body)

        volume = np.where(rth, base_vol * (0.25 + 0.75 * _intraday_vol_tmpl(rth_frac)), base_vol * 0.06)
        volume *= (1.0 + 0.4 * np.abs(rnd))

        df = pd.DataFrame(
            {"open": openp, "high": high, "low": low, "close": c, "volume": volume},
            index=idx,
        ).loc[s:e]
        return normalize_bars(df, symbol)  # type: ignore[return-value]