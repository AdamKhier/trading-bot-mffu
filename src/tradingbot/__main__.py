"""`tradingbot` CLI entry point."""
from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingbot", description="Novel futures strategy research harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="backtest strategy grid (train/OOS) and report")
    run.add_argument("--strategy", default="all", help="all | auction_exhaustion | nq_es_divergence | midday_liquidity | overnight_handoff")
    run.add_argument("--symbols", default=None, help="comma list, default from config (MNQ,MES)")
    run.add_argument("--start", default=None)
    run.add_argument("--end", default=None)
    run.add_argument("--train-end", default=None, help="walk-forward split date (default: 2024-12-31)")
    run.add_argument("--demo", action="store_true", help="use synthetic data (no API key needed)")

    data = sub.add_parser("data", help="data tools")
    dsub = data.add_subparsers(dest="data_cmd", required=True)
    dl = dsub.add_parser("download", help="download continuous 1-min OHLCV from Databento")
    dl.add_argument("--symbols", default="MNQ,MES,NQ,ES")
    dl.add_argument("--start", default=None)
    dl.add_argument("--end", default=None)
    st = dsub.add_parser("status", help="show cached data coverage")
    st.add_argument("--symbols", default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    from tradingbot.config import DEFAULT_CONFIG
    from tradingbot.data.cli import download, status
    from tradingbot.research.runner import run_all, write_reports
    from tradingbot.strategies import STRATEGIES, Strategy

    if args.cmd == "data":
        if args.data_cmd == "download":
            syms = [s.strip() for s in args.symbols.split(",")]
            download(DEFAULT_CONFIG, syms, args.start or DEFAULT_CONFIG.default_start, args.end or DEFAULT_CONFIG.default_end)
        elif args.data_cmd == "status":
            syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
            status(DEFAULT_CONFIG, syms)
        return

    if args.cmd == "run":
        symbols = [s.strip() for s in (args.symbols or ",".join(DEFAULT_CONFIG.symbols)).split(",")]
        results = run_all(DEFAULT_CONFIG, symbols, demo=args.demo, start=args.start, end=args.end, train_end=args.train_end)
        csv, txt = write_reports(results, DEFAULT_CONFIG)
        print(f"\n[reports] CSV: {csv}\n          TXT: {txt}")
        return


if __name__ == "__main__":
    main(sys.argv[1:])