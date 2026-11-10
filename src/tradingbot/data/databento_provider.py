"""Databento historical data provider (continuous front-month futures, 1-min OHLCV)."""
from __future__ import annotations

import pandas as pd

import databento as db

from tradingbot.data.records import normalize_bars
from tradingbot.data.provider import DataProvider


class DatabentoProvider(DataProvider):
    dataset = "GLBX.MDP3"

    def __init__(self, api_key: str):
        if not api_key or api_key == "db-YOUR_KEY_HERE":
            raise ValueError(
                "Databento API key missing. Add DATABENTO_API_KEY=<key> to .env "
                "(sign up at https://databento.com, $125 free credits)."
            )
        self.client = db.Historical(api_key)

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        data = self.client.timeseries.get_range(
            dataset=self.dataset,
            symbols=symbol,
            stype_in="continuous",
            stype_out="raw_symbol",
            schema="ohlcv-1m",
            start=start,
            end=end,
        )
        df = data.to_df(tz="US/Eastern")  # tz-aware timestamps
        return normalize_bars(df, symbol)