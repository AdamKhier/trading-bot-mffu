# Trading Bot — Novel Futures Strategy Research (MFFU evals)

Research harness for discovering and backtesting *under-trodden* intraday futures
strategies on MNQ/MES, with MFFU evaluation rules simulated in the results.

## Status
- [x] Phase 1 environment + data layer (Databento continuous 1-min OHLCV, parquet cache, synthetic fallback)
- [x] Bar-level backtest engine (fees, slippage, risk sizing, session rules)
- [x] MFFU eval simulator (trailing drawdown, profit target, consistency rule)
- [x] 4 novel strategy modules + grid runner (train/out-of-sample split)
- [ ] Real data download (needs your Databento key)
- [ ] First full research sweep + keep/drop checkpoint

## Quick start
```bash
pipx install uv                # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                        # creates .venv with Python 3.12
cp .env.example .env           # add your DATABENTO_API_KEY (https://databento.com → $125 free credits)
uv run tradingbot data download            # real MNQ/MES/NQ/ES 1-min data, 2020→today
uv run tradingbot run --strategy all       # full grid sweep (train/OOS) -> reports/
uv run tradingbot run --strategy all --demo  # smoke test on synthetic data (no key needed)
uv run pytest                   # unit tests
```

## Data
- Source: **Databento** `GLBX.MDP3`, schema `ohlcv-1m`, continuous front-month
  (`stype_in="continuous"`). CME 1-min OHLCV is essentially free; the $125 signup
  credit covers many months of backtests.
- Cached as per-symbol parquet in `data/` for fast repeat runs.
- `--demo` uses `SyntheticProvider` for end-to-end pipeline smoke tests only — **not**
  for discovering edge.

## Strategies (Phase 1 candidates)
| Name | Thesis |
|---|---|
| `auction_exhaustion` | Fade flow-poor stretches: distance-traveled per unit volume far from normal reverts |
| `nq_es_divergence` | Fade statistical richness/cheapness of NQ vs ES residual (stale-beta drift + index arb) |
| `midday_liquidity` | Trade the noon liquidity collapse: thin-regime pin range + expansion breakout |
| `overnight_handoff` | Overnight auction geometry (VWAP position, compressed/expanded range) → open-handoff trade |

## Compliance guardrails (built into the engine)
- Intraday only; positions flat by 15:57 ET (MFFU day-trading behavior).
- No hedging, no DCA/martingale — one instrument, one-sided, fixed risk per trade.
- Conservative risk default: $50 risk/trade, max 3 micros.
- Eval sim enforces trailing drawdown + profit target + consistency rule so a strategy
  is scored on the *actual goal*: pass the eval without breaching.

## Repo layout
```
src/tradingbot/
  config.py              # products, risk, MFFU eval params
  data/                  # provider, databento, synthetic, records (session tagging)
  engine/                # bar-level backtester + metrics
  evalsim/               # MFFU eval simulator
  strategies/            # the 4 strategy modules + grid expansion
  research/run.py        # grid runner + reports
tests/                   # unit tests
```

## Disclaimer
Research tooling only. Automated futures trading involves substantial risk; the
eval simulator approximates MFFU rules and plan parameters must be confirmed against
your actual account before live use.