"""Data provider interface with Parquet cache."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return normalized OHLCV bars (see records.normalize_bars) in [start, end)."""


def cache_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{symbol}_ohlcv_1m.parquet"


class CachedProvider(DataProvider):
    """Wraps an upstream provider with a parquet cache for repeatability."""

    def __init__(self, upstream: DataProvider, data_dir: Path):
        self.upstream = upstream
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = cache_path(self.data_dir, symbol)
        cache = self._read_cache(path, end)
        if cache is not None:
            df = cache
        else:
            df = self.upstream.load(symbol, start, end)
            self._write_cache(path, df)
        mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))
        return df[mask].copy()

    @staticmethod
    def _read_cache(path: Path, end: str) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if len(df) == 0:
            return None
        # Regenerate derived columns in case the tagged calendar rules changed.
        from tradingbot.data.records import normalize_bars

        if "in_rth" not in df.columns or "trade_date" not in df.columns:
            raise ValueError(f"cache {path} is missing derived columns; delete it and retry")
        return df

    @staticmethod
    def _write_cache(path: Path, df: pd.DataFrame) -> None:
        df.to_parquet(path, index=True)