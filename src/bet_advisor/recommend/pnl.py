"""
P&L snapshot utilities for the recommendation engine.

Computes today/week/month/all-time bet counts, units staked and won, ROI with
Wilson confidence interval, mean CLV, and percentage of bets with positive CLV.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bet_advisor.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def compute_pnl_snapshot(
    sqlite_store: SQLiteStore,
    as_of: date,
    bankroll: float,
) -> dict[str, Any]:
    """Compute a P&L snapshot for the bet log as of a given date.

    Parameters
    ----------
    sqlite_store:
        Connected SQLiteStore instance.
    as_of:
        Reference date. "Today" is this date.
    bankroll:
        Current bankroll in currency units (for percentage calculations).

    Returns
    -------
    dict with keys:
        today_bets        -- number of bets placed today
        today_units       -- total stake (currency) placed today
        week_bets         -- bets in the rolling 7 days up to as_of
        week_units        -- total stake this week
        month_bets        -- bets in the current calendar month
        month_units       -- total stake this month
        alltime_bets      -- all bets in the log
        alltime_units     -- total stake all time
        alltime_won_units -- total payout all time
        roi_alltime       -- (won_units - stake_units) / stake_units or None
        roi_wilson_lower  -- Wilson 95% CI lower on win rate
        roi_wilson_upper  -- Wilson 95% CI upper on win rate
        mean_clv          -- mean CLV across settled bets with clv_pct (or None)
        pct_positive_clv  -- fraction of positive-CLV bets (or None)
        n_settled         -- number of settled bets with CLV data
    """
    today_str = as_of.isoformat()
    week_start = (as_of - timedelta(days=6)).isoformat()
    month_start = as_of.replace(day=1).isoformat()

    # All bets
    all_bets = sqlite_store.query("SELECT * FROM bets")

    today_bets = [b for b in all_bets if b.get("placed_at", "")[:10] == today_str]
    week_bets = [b for b in all_bets if b.get("placed_at", "")[:10] >= week_start]
    month_bets = [b for b in all_bets if b.get("placed_at", "")[:10] >= month_start]

    def _sum_stake(rows: list[dict]) -> float:
        return sum(float(r.get("stake") or 0) for r in rows)

    def _sum_payout(rows: list[dict]) -> float:
        return sum(float(r.get("payout") or 0) for r in rows if r.get("payout") is not None)

    alltime_units = _sum_stake(all_bets)
    alltime_won_units = _sum_payout(all_bets)

    roi_alltime: float | None = None
    if alltime_units > 0:
        roi_alltime = (alltime_won_units - alltime_units) / alltime_units

    # Win rate and Wilson CI
    settled = [b for b in all_bets if b.get("status") in ("won", "lost")]
    n_settled_outcomes = len(settled)
    n_won = sum(1 for b in settled if b.get("status") == "won")

    wilson_lower: float | None = None
    wilson_upper: float | None = None
    if n_settled_outcomes > 0:
        wilson_lower, wilson_upper = _wilson_ci(n_won, n_settled_outcomes)

    # CLV stats
    clv_bets = [b for b in all_bets if b.get("clv_pct") is not None]
    n_clv = len(clv_bets)
    mean_clv: float | None = None
    pct_positive_clv: float | None = None
    if n_clv > 0:
        clv_values = [float(b["clv_pct"]) for b in clv_bets]
        mean_clv = sum(clv_values) / n_clv
        pct_positive_clv = sum(1 for v in clv_values if v > 0) / n_clv

    return {
        "today_bets": len(today_bets),
        "today_units": _sum_stake(today_bets),
        "week_bets": len(week_bets),
        "week_units": _sum_stake(week_bets),
        "month_bets": len(month_bets),
        "month_units": _sum_stake(month_bets),
        "alltime_bets": len(all_bets),
        "alltime_units": alltime_units,
        "alltime_won_units": alltime_won_units,
        "roi_alltime": roi_alltime,
        "roi_wilson_lower": wilson_lower,
        "roi_wilson_upper": wilson_upper,
        "mean_clv": mean_clv,
        "pct_positive_clv": pct_positive_clv,
        "n_settled": n_clv,
        "bankroll": bankroll,
    }


def _wilson_ci(n_positive: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    import math

    if n_total == 0:
        return 0.0, 1.0

    # z for 95% CI
    z = 1.959964
    p_hat = n_positive / n_total
    denominator = 1.0 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denominator
    spread = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)
