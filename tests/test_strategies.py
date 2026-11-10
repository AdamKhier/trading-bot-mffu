"""Strategy smoke tests: each strategy builds a well-formed signal frame on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingbot.data.synthetic import SyntheticProvider
from tradingbot.strategies import STRATEGIES, get_strategy


@pytest.fixture(scope="module")
def bars():
    p = SyntheticProvider()
    return {
        "MNQ": p.load("MNQ", "2021-01-01", "2021-06-01"),
        "MES": p.load("MES", "2021-01-01", "2021-06-01"),
    }


BARS_PER_COMBO = {
    "auction_exhaustion": 4,
    "nq_es_divergence": 6,
    "midday_liquidity": 6,
    "overnight_handoff": 4,
}


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_signals_aligned_and_clean(name, bars):
    cls = STRATEGIES[name]
    for combo in cls.expand_grid():
        strat = get_strategy(name, combo)
        if name == "nq_es_divergence" and "MES" not in bars:
            continue
        df = strat.build_signals(bars)
        primary = strat.params.get("leg", "MNQ") if name == "nq_es_divergence" else "MNQ"
        assert list(df.index) == list(bars[primary].index), f"{name}: index mismatch vs {primary}"
        for col in ("signal", "stop_dist_ticks", "target_dist_ticks"):
            assert col in df.columns, f"{name}: missing {col}"
        assert set(df["signal"].unique()).issubset({-1, 0, 1})
        assert df["stop_dist_ticks"].dropna().ge(1).all()
        assert df["target_dist_ticks"].dropna().ge(1).all()
        if "valid" in df.columns:
            assert df["valid"].dropna().isin([True, False]).all()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_produces_some_trades_after_backtest(name, bars):
    from tradingbot.config import DEFAULT_CONFIG
    from tradingbot.engine import compute_metrics, run_backtest

    cls = STRATEGIES[name]
    combos = cls.expand_grid()
    strat = get_strategy(name, combos[len(combos) // 2])
    df = strat.build_signals(bars)
    primary = strat.params.get("leg", "MNQ") if name == "nq_es_divergence" else "MNQ"
    bb = bars[primary]
    sig = df.reindex(bb.index)
    res = run_backtest(
        bb,
        sig,
        primary,
        risk=DEFAULT_CONFIG.risk,
        flat_time_et=DEFAULT_CONFIG.eval.flat_time_et,
        max_trades_per_day=getattr(cls, "max_trades_per_day", 2),
    )
    m = compute_metrics(res.trades, res.daily_pnl)
    assert m.n_trades >= 1, f"{name}: expected at least one trade over 6m synthetic"
    assert m.n_days >= 1