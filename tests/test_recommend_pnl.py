"""
Tests for compute_pnl_snapshot.

Uses a synthetic bet log in an in-memory SQLite store.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from bet_advisor.recommend.pnl import compute_pnl_snapshot
from bet_advisor.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_with_bets(tmp_path: Path) -> SQLiteStore:
    """SQLite store seeded with a mix of bets across dates."""
    db_path = tmp_path / "pnl_test.db"
    store = SQLiteStore(db_path)
    store.connect()

    # Create a placeholder event for FK constraint
    store.upsert_event(
        event_id="evt-pnl",
        sport_key="aussierules_afl",
        sport_title="AFL",
        commence_time="2026-04-01T19:30:00",
        home_team="Collingwood",
        away_team="Hawthorn",
    )

    today = date(2026, 5, 15)
    # Insert signal first
    signal_id = store.insert_signal(
        event_id="evt-pnl",
        market="h2h",
        runner="Collingwood",
        model_prob=0.60,
        market_prob_devigged=0.53,
        edge=0.05,
        recommended_stake=10.0,
        model_version="v1",
        created_at="2026-04-01T00:00:00",
    )

    # Insert bets spread across today, this week, this month, older
    bet_configs = [
        # today -- two bets
        {"placed_at": f"{today.isoformat()}T10:00:00", "stake": 10.0, "payout": 18.0, "status": "won", "clv_pct": 0.02},
        {"placed_at": f"{today.isoformat()}T11:00:00", "stake": 10.0, "payout": None, "status": "lost", "clv_pct": -0.01},
        # this week (3 days ago)
        {"placed_at": f"{(today - timedelta(days=3)).isoformat()}T10:00:00", "stake": 10.0, "payout": 18.0, "status": "won", "clv_pct": 0.03},
        # this month (10 days ago)
        {"placed_at": f"{(today - timedelta(days=10)).isoformat()}T10:00:00", "stake": 10.0, "payout": None, "status": "lost", "clv_pct": None},
        # older (last month)
        {"placed_at": "2026-04-01T10:00:00", "stake": 10.0, "payout": 18.0, "status": "won", "clv_pct": 0.04},
    ]

    for cfg in bet_configs:
        bet_id = store.log_bet(
            signal_id=signal_id,
            placed_at=cfg["placed_at"],
            bookmaker="sportsbet",
            market="h2h",
            runner="Collingwood",
            price=1.90,
            stake=cfg["stake"],
        )
        if cfg["status"] in ("won", "lost"):
            store.settle_bet(
                bet_id=bet_id,
                status=cfg["status"],
                payout=cfg["payout"],
                closing_price=1.88,
                clv_pct=cfg["clv_pct"],
                settled_at=cfg["placed_at"],
            )

    yield store
    store.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputePnlSnapshot:
    def test_today_bet_count(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        assert snap["today_bets"] == 2

    def test_today_units_correct(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        assert snap["today_units"] == pytest.approx(20.0)

    def test_week_bets_correct(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        # today (2) + 3 days ago (1) = 3
        assert snap["week_bets"] == 3

    def test_month_bets_excludes_older(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        # May bets only: today 2 + 3 days ago 1 + 10 days ago 1 = 4
        assert snap["month_bets"] == 4

    def test_alltime_bets_includes_all(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        assert snap["alltime_bets"] == 5

    def test_alltime_won_units_correct(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        # Three bets won: payout 18 each = 54
        assert snap["alltime_won_units"] == pytest.approx(54.0)

    def test_roi_computed_when_bets_exist(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        assert snap["roi_alltime"] is not None
        # Total stake = 50, payout = 54, ROI = (54 - 50) / 50 = 0.08
        assert snap["roi_alltime"] == pytest.approx(0.08)

    def test_mean_clv_computed(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        # CLV values: 0.02, -0.01, 0.03, 0.04 (null excluded) -- mean = (0.02 - 0.01 + 0.03 + 0.04) / 4 = 0.02
        assert snap["mean_clv"] == pytest.approx(0.02)

    def test_pct_positive_clv(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        # 3 positive (0.02, 0.03, 0.04) out of 4 with clv
        assert snap["pct_positive_clv"] == pytest.approx(0.75)

    def test_wilson_ci_present(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=1000.0)
        assert snap["roi_wilson_lower"] is not None
        assert snap["roi_wilson_upper"] is not None
        assert snap["roi_wilson_lower"] <= snap["roi_wilson_upper"]

    def test_bankroll_in_snapshot(self, store_with_bets: SQLiteStore) -> None:
        snap = compute_pnl_snapshot(store_with_bets, as_of=date(2026, 5, 15), bankroll=5000.0)
        assert snap["bankroll"] == pytest.approx(5000.0)

    def test_empty_store_returns_zeros(self, tmp_path: Path) -> None:
        db = SQLiteStore(tmp_path / "empty.db")
        db.connect()
        try:
            snap = compute_pnl_snapshot(db, as_of=date(2026, 5, 15), bankroll=1000.0)
            assert snap["alltime_bets"] == 0
            assert snap["roi_alltime"] is None
            assert snap["mean_clv"] is None
        finally:
            db.close()
