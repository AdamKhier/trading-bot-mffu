"""`tradingbot data` commands: download / status."""
from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

from tradingbot.config import DEFAULT_CONFIG, Config
from tradingbot.data.databento_provider import DatabentoProvider
from tradingbot.data.provider import CachedProvider, cache_path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = lambda *a, **k: None  # type: ignore[assignment]


def _key() -> str:
    load_dotenv(DEFAULT_CONFIG.data_dir.parent / ".env")
    load_dotenv()
    return os.environ.get(DEFAULT_CONFIG.api_key_env, "")


def _provider(config: Config) -> CachedProvider:
    return CachedProvider(DatabentoProvider(_key()), config.data_dir)


def download(config: Config, symbols: list[str], start: str, end: str) -> None:
    prov = _provider(config)
    for sym in symbols:
        print(f"downloading {sym} [{start} -> {end}] ...")
        df = prov.load(sym, start, end)
        n = len(df)
        rth = int(df["in_rth"].sum())
        lo, hi = (df.index.min(), df.index.max()) if n else (None, None)
        print(f"  {sym}: {n:,} bars  (RTH {rth:,})  {lo} -> {hi}")


def status(config: Config, symbols: list[str] | None = None) -> None:
    syms = symbols or list(config.symbols)
    any_missing = False
    for sym in syms:
        p = cache_path(config.data_dir, sym)
        if p.exists():
            df = pd.read_parquet(p)
            lo, hi = df.index.min(), df.index.max()
            print(f"  {sym:>4}: CACHED  {len(df):,} bars  {lo} -> {hi}")
        else:
            print(f"  {sym:>4}: MISSING (run `tradingbot data download`)")
            any_missing = True
    if any_missing:
        print("Tip: use `--demo` with `run` to smoke-test on synthetic data without a key.")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="tradingbot data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="download continuous 1-min OHLCV from Databento")
    dl.add_argument("--symbols", default="MNQ,MES,NQ,ES")
    dl.add_argument("--start", default="2020-01-01")
    dl.add_argument("--end", default=str(date.today().isoformat()))

    st = sub.add_parser("status", help="show cached data coverage")
    st.add_argument("--symbols", default=None)

    args = parser.parse_args(argv)
    config = DEFAULT_CONFIG

    if args.cmd == "download":
        download(config, [s.strip() for s in args.symbols.split(",")], args.start, args.end)
    elif args.cmd == "status":
        syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
        status(config, syms)


if __name__ == "__main__":
    main(sys.argv[1:])