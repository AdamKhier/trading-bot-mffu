"""Grid runner: backtest every strategy/param combo with train/OOS split + eval sim."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tradingbot.config import Config, DEFAULT_CONFIG
from tradingbot.data.provider import CachedProvider
from tradingbot.data.synthetic import SyntheticProvider
from tradingbot.engine import compute_metrics, run_backtest
from tradingbot.evalsim import simulate_eval
from tradingbot.strategies import STRATEGIES, Strategy


@dataclass
class ComboResult:
    strategy: str
    display: str
    symbol: str
    params: dict
    train: dict = field(default_factory=dict)
    oos: dict = field(default_factory=dict)
    rank_score: float = 0.0


def _load_databento_key(config: Config) -> str:
    from dotenv import load_dotenv

    load_dotenv(config.data_dir.parent / ".env")
    load_dotenv()
    return __import__("os").environ.get(config.api_key_env, "")


def load_bars(config: Config, symbols: list[str], start: str, end: str, demo: bool = False) -> dict[str, pd.DataFrame]:
    if demo:
        print("[data] using SYNTHETIC data (smoke test only - not for edge discovery)")
        prov = SyntheticProvider()
        return {s: prov.load(s, start, end) for s in symbols}
    from tradingbot.data.databento_provider import DatabentoProvider

    key = _load_databento_key(config)
    if not key or key == "db-YOUR_KEY_HERE":
        raise SystemExit("Databento key not configured. Add DATABENTO_API_KEY to .env, or run with --demo to use synthetic data.")
    cached = CachedProvider(DatabentoProvider(key), config.data_dir)
    return {s: cached.load(s, start, end) for s in symbols}


def run_combo(
    strategy: Strategy,
    cls_params_max_trades: int,
    bars: dict[str, pd.DataFrame],
    start: str,
    end: str,
    config: Config,
) -> ComboResult:
    primary = strategy.params.get("leg", list(bars)[0])
    sig = strategy.build_signals(bars)
    if primary not in bars:
        primary = list(bars)[0]
    symbol = primary

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    train_hi = pd.Timestamp(config.train_end, tz="UTC")
    train_bars = bars[symbol].loc[start_ts:train_hi]
    train_sig = sig.reindex(train_bars.index)
    oos_bars = bars[symbol].loc[train_hi:end_ts]
    oos_sig = sig.reindex(oos_bars.index)

    def run(bars_: pd.DataFrame, sig_: pd.DataFrame) -> dict:
        res = run_backtest(
            bars_,
            sig_,
            symbol,
            risk=config.risk,
            flat_time_et=config.eval.flat_time_et,
            max_trades_per_day=cls_params_max_trades,
        )
        metrics = compute_metrics(res.trades, res.daily_pnl)
        ev = simulate_eval(res.daily_pnl, config.eval)
        d = metrics.as_dict()
        d["eval_passed"] = ev.passed
        d["eval_breached"] = ev.breached
        d["eval_days_to_pass"] = ev.days_to_pass
        d["eval_max_dd"] = round(ev.max_drawdown, 2)
        d["eval_consistency_ok"] = ev.consistency_ok
        d["eval_note"] = ev.note
        d["n_trades"] = metrics.n_trades
        d["net_pnl"] = round(metrics.net_pnl, 2)
        return d

    cr = ComboResult(strategy=strategy.name, display=strategy.display_name, symbol=symbol, params=strategy.params)
    cr.train = run(train_bars.dropna(), train_sig) if len(train_bars) else {}
    cr.oos = run(oos_bars.dropna(), oos_sig) if len(oos_bars) else {}
    cr.rank_score = _rank_score(cr)
    return cr


def _rank_score(cr: ComboResult) -> float:
    """Simple ranking: OOS net pnl with a drawdown penalty and breaching disqualifier."""
    m = cr.oos
    if not m or m.get("eval_breached", False):
        return float("-inf")
    pnl = m.get("net_pnl", 0.0)
    dd_pen = max(m.get("max_dd_usd", 0.0) / 2000.0, 0.0)
    return pnl - dd_pen * 5000.0


def run_all(config: Config, symbols: list[str], demo: bool = False, start: str | None = None, end: str | None = None, train_end: str | None = None) -> list[ComboResult]:
    from copy import deepcopy

    config = deepcopy(config)
    if train_end:
        config.train_end = train_end
    start = start or config.default_start
    end = end or config.default_end
    print(f"[data] loading bars [{start} -> {end}] symbols={symbols} demo={demo}")
    bars = load_bars(config, symbols, start, end, demo=demo)
    for s, df in bars.items():
        print(f"  {s}: {len(df):,} bars")

    results: list[ComboResult] = []
    for name, cls in STRATEGIES.items():
        strat = cls({})
        need = getattr(strat, "requires", lambda: [])()
        if not all(sym in bars for sym in need):
            print(f"  skip {name}: requires {need}")
            continue
        max_trades = getattr(cls, "max_trades_per_day", 2)
        print(f"[run] {name} ({cls.expand_grid()} combos)")
        for combo in cls.expand_grid():
            try:
                s = cls(combo)
                res = run_combo(s, max_trades, bars, start, end, config)
                results.append(res)
            except Exception as exc:  # pragma: no cover
                print(f"  !! {name} {combo}: {exc}")
    return results


def write_reports(results: list[ComboResult], config: Config) -> tuple[Path, Path]:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        base = {
            "strategy": r.strategy,
            "display": r.display,
            "symbol": r.symbol,
            "params": json.dumps(r.params),
            "rank_score": round(r.rank_score, 2),
        }
        row = {**base}
        for prefix in ("train", "oos"):
            for k, v in r.__dict__[prefix].items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = config.report_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)

    best = sorted(results, key=lambda r: r.rank_score, reverse=True)[:10]
    txt = _format_summary(results, best)
    txt_path = config.report_dir / "summary.txt"
    txt_path.write_text(txt)
    print(txt)
    return csv_path, txt_path


def _format_summary(results: list[ComboResult], best: list[ComboResult]) -> str:
    from collections import Counter

    lines = ["=== Strategy Research Summary ===", ""]
    for name in STRATEGIES:
        rs = [r for r in results if r.strategy == name]
        if not rs:
            continue
        oos_pnl = [r.oos.get("net_pnl", 0.0) for r in rs]
        oos_breached = sum(1 for r in rs if r.oos.get("eval_breached"))
        lines.append(f"[{name}] combos={len(rs)}  OOS net PnL {sum(oos_pnl):,.0f}  breached {oos_breached}")
    lines.append("")
    lines.append("TOP 10 (by OOS net PnL w/ drawdown penalty):")
    for r in best:
        key = json.dumps(r.params)
        lines.append(
            f"  {r.symbol:>4} {r.strategy:<20} {key:<42} train={r.train.get('net_pnl', 0):>9,.0f} "
            f"oos={r.oos.get('net_pnl', 0):>9,.0f} trades={r.oos.get('n_trades', 0)} "
            f"wr={r.oos.get('win_rate', 0):.2f} dd={r.oos.get('max_dd_usd', 0):,.0f} "
            f"evar={r.oos.get('eval_passed')}"
        )
    return "\n".join(lines)