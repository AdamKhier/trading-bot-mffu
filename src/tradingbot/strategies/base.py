"""Strategy interface and common helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product as cartesian

import pandas as pd


class Strategy(ABC):
    name: str
    display_name: str
    grid_keys: list[str] = []
    default_params: dict = {}

    def __init__(self, params: dict):
        self.params = {**self.default_params, **params}

    @abstractmethod
    def build_signals(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a signal frame aligned to the primary bars index.

        Required columns:
          signal            int   {-1, 0, +1}  target position at this bar's close
          stop_dist_ticks   float distance in ticks from entry to stop (>= 1)
          target_dist_ticks float distance in ticks from entry to target (>= 1)
        Optional columns:
          valid       bool  entries allowed
          hard_exit   bool  exit at this bar's close regardless of stop/target
        """

    @classmethod
    def expand_grid(cls) -> list[dict]:
        if not cls.grid_keys:
            return [{}]
        vals = [cls.default_params[k] for k in cls.grid_keys]
        return [dict(zip(cls.grid_keys, combo)) for combo in cartesian(*vals)]


def zscore_ts(s: pd.Series, win: int) -> pd.Series:
    """Centered z-score with expanding-rolling std."""
    mu = s.rolling(win, min_periods=max(win // 2, 10)).mean()
    sig = s.rolling(win, min_periods=max(win // 2, 10)).std()
    return (s - mu) / (sig + 1e-12)


def ema_beta(y: pd.Series, x: pd.Series, half_life_min: int) -> pd.Series:
    """Exponentially-weighted regression coefficient beta = cov(y,x)/var(x)."""
    alpha = 1.0 - 2 ** (-1.0 / half_life_min)
    cov = (y * x).ewm(alpha=alpha, adjust=False).mean()
    var = (x * x).ewm(alpha=alpha, adjust=False).mean()
    return cov / (var + 1e-12)