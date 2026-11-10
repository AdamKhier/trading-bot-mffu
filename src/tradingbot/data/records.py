"""Bar normalization: standard OHLCV frame + session/trade-date tagging."""

from __future__ import annotations

import pandas as pd

from tradingbot.config import EASTERN, GLOBEX_OPEN_ET, RTH_CLOSE_ET, RTH_OPEN_ET, UTC


def time_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a tz-aware index to US/Eastern wall-clock."""
    return index.tz_convert(EASTERN)  # type: ignore[arg-type]


def normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize a Databento ohlcv-1m frame into the standard research schema.

    Returns a DataFrame indexed by UTC timestamps with columns:
      open, high, low, close, volume, symbol,
      time_et (datetime[US/Eastern]), in_rth (bool), trade_date (date, ET business day)
    """
    if df is None or df.empty:
        raise ValueError(f"empty bars for {symbol}")

    df = df.sort_index()
    if isinstance(df.index, pd.RangeIndex) and "ts_event" in df.columns:
        df = df.set_index(pd.to_datetime(df["ts_event"], unit="ns")).sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if len(cols) != 5:
        raise ValueError(f"bars for {symbol} missing OHLCV columns; got {list(df.columns)}")
    df = df[cols].copy()

    idx = pd.DatetimeIndex(df.index)
    if not idx.tz:
        idx = idx.tz_localize(UTC)
    df.index = idx.tz_convert("UTC")
    df.index.name = "ts"

    et = time_et(df.index)
    df["time_et"] = et
    df["in_rth"] = _rth_mask(et)
    df["trade_date"] = _trade_date(et)
    df["symbol"] = symbol
    return df


def _rth_mask(et: pd.DatetimeIndex) -> pd.Series:
    t = et.strftime("%H:%M")
    return pd.Series((t >= RTH_OPEN_ET) & (t < RTH_CLOSE_ET) & (et.dayofweek < 5), index=et)


def _trade_date(et: pd.DatetimeIndex) -> pd.Series:
    """Assign an ET business date. Globex prior-evening bars (18:00-24:00 EST) roll to the next day."""
    dates = et.normalize()
    roll = (et.hour >= int(GLOBEX_OPEN_ET.split(":")[0])) & (dates.dayofweek < 5)
    rolled = dates + pd.to_timedelta(roll.astype(int), unit="D")
    return rolled.strftime("%Y-%m-%d")


def rth_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["in_rth"]].copy()