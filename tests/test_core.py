"""Unit tests for the backtest engine, metrics, eval sim, and data plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.config import MFFUEvalParams, RiskParams
from tradingbot.data.records import normalize_bars
from tradingbot.data.synthetic import SyntheticProvider
from tradingbot.engine import Trade, compute_metrics, run_backtest
from tradingbot.evalsim import simulate_eval


def make_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def test_normalize_bars_tags_sessions():
    df = SyntheticProvider().load("MNQ", "2021-01-04", "2021-01-08")
    rth = df[df["in_rth"]]
    assert len(rth) == len(set(rth["trade_date"])) * 390
    assert set(["open", "high", "low", "close", "volume", "time_et", "in_rth", "trade_date", "symbol"]).issubset(
        df.columns
    )
    assert df["trade_date"].notna().all()


def _handcrafted_day() -> pd.DataFrame:
    """Deterministic rising day: closes climb 0.5/min, low never retraces far."""
    n = 120
    idx = pd.date_range("2021-03-01 09:30", periods=n, freq="min", tz="US/Eastern").tz_convert("UTC")
    closes = 12_000.0 + np.arange(n) * 0.5
    opens = closes - 0.2
    highs = closes + 0.3
    lows = closes - 0.3
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.full(n, 400)}, index=idx)
    return normalize_bars(df, "MNQ")


def test_backtester_long_profitable_trade():
    bars = _handcrafted_day()
    sig = pd.DataFrame(index=bars.index)
    sig["signal"] = 0
    sig["stop_dist_ticks"] = 20.0
    sig["target_dist_ticks"] = 10.0
    sig["valid"] = False
    sig["hard_exit"] = False
    sig.loc[bars.index[0], "signal"] = 1
    sig.loc[bars.index[0], "valid"] = True

    res = run_backtest(bars, sig, "MNQ", risk=RiskParams(risk_per_trade=100, max_contracts=1), flat_time_et="15:57")
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.side == 1
    assert t.exit_reason == "target"
    assert t.pnl > 0
    assert "MNQ" in res.symbol


def test_backtester_forces_flat():
    bars = _handcrafted_day()
    sig = pd.DataFrame(index=bars.index)
    sig["signal"] = 0
    sig["stop_dist_ticks"] = 100000.0
    sig["target_dist_ticks"] = 1e9
    sig["valid"] = False
    sig["hard_exit"] = False
    sig.loc[bars.index[0], "signal"] = 1
    sig.loc[bars.index[0], "valid"] = True
    res = run_backtest(bars, sig, "MNQ", risk=RiskParams(risk_per_trade=100, max_contracts=1), flat_time_et="10:00")
    assert res.trades[-1].exit_reason == "time"
    assert res.trades[-1].exit_ts.tz_convert("US/Eastern").hour == 10  # flat at 10:00 bar close


def test_metrics_basics():
    t1 = Trade("MNQ", 1, 1, 100.0, 110.0, pd.Timestamp("2021-01-04T09:31Z"), pd.Timestamp("2021-01-04T09:40Z"), "target", 20.0, 2.46, 9)
    t2 = Trade("MNQ", -1, 1, 100.0, 95.0, pd.Timestamp("2021-01-04T10:00Z"), pd.Timestamp("2021-01-04T10:10Z"), "stop", -10.0, 2.46, 10)
    daily = pd.Series([20.0, -10.0], index=pd.Index(["2021-01-04", "2021-01-05"]))
    m = compute_metrics([t1, t2], daily)
    assert m.n_trades == 2
    assert m.net_pnl == 10.0
    assert m.win_rate == 0.5
    assert m.profit_factor > 1
    assert m.max_dd_usd < 0 or m.max_dd_usd == 0


def test_eval_pass_when_target_without_breach():
    pnl = pd.Series([500.0] * 8, index=pd.date_range("2021-01-04", periods=8))
    res = simulate_eval(pnl, MFFUEvalParams(profit_target=3000.0, max_loss=2000.0, min_trading_days=4, consistency_factor=0.5))
    assert res.passed
    assert not res.breached
    assert res.days_to_pass == 6


def test_eval_breaches_max_dd():
    pnl = pd.Series([1000.0, -2500.0, 3000.0], index=pd.date_range("2021-01-04", periods=3))
    res = simulate_eval(pnl, MFFUEvalParams(profit_target=3000.0, max_loss=2000.0, min_trading_days=3, consistency_factor=0.5))
    assert res.breached
    assert not res.passed


def test_independent_days_are_stable():
    d1 = SyntheticProvider().load("MNQ", "2021-01-04", "2021-01-05")
    d2 = SyntheticProvider().load("MNQ", "2021-01-04", "2021-01-05")
    pd.testing.assert_frame_equal(normalize_bars(d1, "MNQ"), normalize_bars(d2, "MNQ"))