"""Simulate whether a strategy would pass an MFFU evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingbot.config import MFFUEvalParams


@dataclass
class EvalSimResult:
    passed: bool
    breached: bool
    days_to_pass: int | None
    breach_date: object | None
    final_pnl: float
    max_drawdown: float
    consistency_ok: bool | None  # None when consistency not configured
    note: str = ""


def simulate_eval(daily_pnl: pd.Series, params: MFFUEvalParams, account_start: float = 0.0) -> EvalSimResult:
    """Run an equity curve through the eval rulebook.

    ``account_start`` is the equity baseline (typically 0 = starting balance).
    ``daily_pnl`` is indexed by (ET) trade date.
    """
    if len(daily_pnl) == 0:
        return EvalSimResult(False, False, None, None, 0.0, 0.0, None, "no data")

    eq = daily_pnl.cumsum() + account_start
    # synthetic day-0 baseline so the first day's P&L is counted
    eq.index = pd.to_datetime(eq.index)
    eq0 = pd.Series([account_start], index=[eq.index[0] - pd.Timedelta(days=1)])
    eq_full = pd.concat([eq0, eq]).sort_index()

    if params.drawdown_type == "intraday_trailing":
        # per-day close proxy: compare running high watermark vs end-of-day equity
        peak = eq_full.cummax()
        dd = peak - eq_full
        breach_mask = dd > params.max_loss
    else:  # "eod_trailing"
        peak = eq_full.cummax()
        dd = peak - eq_full
        breach_mask = dd > params.max_loss

    breach_idx = list(breach_mask[breach_mask].index)
    breached = len(breach_idx) > 0
    breach_date = breach_idx[0] if breached else None

    # cap at the last full day available
    target_hit = eq.to_numpy() >= (account_start + params.profit_target)
    days_to_target = None
    for k, hit in enumerate(target_hit):
        if hit:
            days_to_target = k + 1
            break

    consistency_ok = None
    if params.consistency_factor and params.consistency_factor > 0:
        wins = daily_pnl[daily_pnl > 0]
        total_win = wins.sum()
        if total_win > 1e-9:
            best = wins.max()
            consistency_ok = (best / total_win) <= params.consistency_factor
        else:
            consistency_ok = None

    consistency_met = (consistency_ok is None) or consistency_ok
    passed = (
        (not breached)
        and (days_to_target is not None)
        and (days_to_target >= params.min_trading_days)
        and consistency_met
    )

    # note about days requirement
    note = ""
    if breached:
        note = f"breached max loss ({params.max_loss:.0f} USD) on {breach_date}"
    elif days_to_target is not None and days_to_target < params.min_trading_days:
        note = "hit target but below minimum trading days"
    elif not consistency_met:
        note = "target hit but best-day consistency rule failed"

    max_dd = float((peak - eq_full).max()) if len(peak) else 0.0
    return EvalSimResult(
        passed=passed,
        breached=breached,
        days_to_pass=days_to_target if (passed or days_to_target is not None) else None,
        breach_date=breach_date,
        final_pnl=float(eq.iloc[-1] - account_start),
        max_drawdown=max_dd,
        consistency_ok=consistency_ok,
        note=note,
    )