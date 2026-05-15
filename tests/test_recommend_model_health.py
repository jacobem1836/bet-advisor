"""
Tests for compute_model_health and trigger flag logic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from bet_advisor.recommend.model_health import (
    _compute_drawdown,
    compute_model_health,
    ensure_model_health_table,
    record_model_health,
)
from bet_advisor.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_store(tmp_path: Path) -> SQLiteStore:
    db_path = tmp_path / "health_test.db"
    store = SQLiteStore(db_path)
    store.connect()
    ensure_model_health_table(store)
    yield store
    store.close()


@pytest.fixture()
def store_with_health(empty_store: SQLiteStore) -> SQLiteStore:
    """Store with a model health snapshot and some bets."""
    # Insert event for FK
    empty_store.upsert_event(
        event_id="evt-h",
        sport_key="aussierules_afl",
        sport_title="AFL",
        commence_time="2026-04-01T19:30:00",
        home_team="A",
        away_team="B",
    )
    sig_id = empty_store.insert_signal(
        event_id="evt-h",
        market="h2h",
        runner="A",
        model_prob=0.60,
        market_prob_devigged=0.53,
        edge=0.05,
        recommended_stake=10.0,
        model_version="v1",
        created_at="2026-04-01T00:00:00",
    )
    # Bets: won, lost, pending
    empty_store.log_bet(
        signal_id=sig_id,
        placed_at="2026-04-01T10:00:00",
        bookmaker="sportsbet",
        market="h2h",
        runner="A",
        price=1.90,
        stake=10.0,
    )

    record_model_health(
        empty_store,
        model_version="v1",
        brier=0.2300,
        log_loss=0.5500,
        ece=0.0150,
        mean_clv=0.012,
    )
    return empty_store


# ---------------------------------------------------------------------------
# Tests: ensure_model_health_table
# ---------------------------------------------------------------------------


class TestEnsureTable:
    def test_creates_table(self, empty_store: SQLiteStore) -> None:
        tables = [r["name"] for r in empty_store.query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        assert "model_health" in tables

    def test_idempotent(self, empty_store: SQLiteStore) -> None:
        # Should not raise on second call
        ensure_model_health_table(empty_store)
        ensure_model_health_table(empty_store)


# ---------------------------------------------------------------------------
# Tests: record_model_health
# ---------------------------------------------------------------------------


class TestRecordModelHealth:
    def test_returns_positive_id(self, empty_store: SQLiteStore) -> None:
        row_id = record_model_health(empty_store, model_version="v1", brier=0.25)
        assert row_id > 0

    def test_stores_brier(self, empty_store: SQLiteStore) -> None:
        record_model_health(empty_store, model_version="v1", brier=0.2500)
        rows = empty_store.query("SELECT brier FROM model_health")
        assert rows[0]["brier"] == pytest.approx(0.2500)

    def test_stores_ece(self, empty_store: SQLiteStore) -> None:
        record_model_health(empty_store, model_version="v1", ece=0.0180)
        rows = empty_store.query("SELECT ece FROM model_health")
        assert rows[0]["ece"] == pytest.approx(0.0180)


# ---------------------------------------------------------------------------
# Tests: compute_model_health
# ---------------------------------------------------------------------------


class TestComputeModelHealth:
    def test_returns_dict_with_expected_keys(self, store_with_health: SQLiteStore) -> None:
        health = compute_model_health(store_with_health, duckdb_store=None)
        required_keys = {
            "last_model_version",
            "last_captured_at",
            "days_since_snapshot",
            "latest_brier",
            "latest_ece",
            "latest_log_loss",
            "brier_two_months_ago",
            "brier_deteriorated",
            "drawdown_pct",
            "ece_trigger",
            "drawdown_trigger",
            "clv_negative_trigger",
            "n_bets",
            "trigger_summary",
        }
        assert required_keys.issubset(health.keys())

    def test_latest_brier_returned(self, store_with_health: SQLiteStore) -> None:
        health = compute_model_health(store_with_health, duckdb_store=None)
        assert health["latest_brier"] == pytest.approx(0.2300)

    def test_latest_ece_returned(self, store_with_health: SQLiteStore) -> None:
        health = compute_model_health(store_with_health, duckdb_store=None)
        assert health["latest_ece"] == pytest.approx(0.0150)

    def test_ece_trigger_false_when_below_threshold(self, store_with_health: SQLiteStore) -> None:
        # ECE 0.0150 < 0.02
        health = compute_model_health(store_with_health, duckdb_store=None)
        assert health["ece_trigger"] is False

    def test_ece_trigger_true_when_above_threshold(self, empty_store: SQLiteStore) -> None:
        record_model_health(empty_store, model_version="v1", ece=0.0300)
        health = compute_model_health(empty_store, duckdb_store=None)
        assert health["ece_trigger"] is True

    def test_no_snapshot_returns_none_version(self, empty_store: SQLiteStore) -> None:
        health = compute_model_health(empty_store, duckdb_store=None)
        assert health["last_model_version"] is None

    def test_trigger_summary_no_triggers(self, store_with_health: SQLiteStore) -> None:
        health = compute_model_health(store_with_health, duckdb_store=None)
        assert health["trigger_summary"] == "No active triggers"

    def test_trigger_summary_with_ece_trigger(self, empty_store: SQLiteStore) -> None:
        record_model_health(empty_store, model_version="v1", ece=0.0500)
        health = compute_model_health(empty_store, duckdb_store=None)
        assert "ECE" in health["trigger_summary"]
        assert health["ece_trigger"] is True

    def test_drawdown_trigger_false_below_25pct(self, store_with_health: SQLiteStore) -> None:
        health = compute_model_health(store_with_health, duckdb_store=None)
        assert health["drawdown_trigger"] is False


# ---------------------------------------------------------------------------
# Tests: _compute_drawdown
# ---------------------------------------------------------------------------


class TestComputeDrawdown:
    def test_zero_drawdown_monotonically_increasing(self) -> None:
        bets = [
            {"stake": 10.0, "payout": 18.0, "status": "won"},
            {"stake": 10.0, "payout": 18.0, "status": "won"},
            {"stake": 10.0, "payout": 18.0, "status": "won"},
        ]
        assert _compute_drawdown(bets) == pytest.approx(0.0)

    def test_drawdown_computed_correctly(self) -> None:
        # Win 8 (net), then lose 10, then win 8
        bets = [
            {"stake": 10.0, "payout": 18.0, "status": "won"},   # cumulative = 8, peak = 8
            {"stake": 10.0, "payout": None, "status": "lost"},   # cumulative = -2, peak = 8, drawdown = 10/8 = 125%
            {"stake": 10.0, "payout": 18.0, "status": "won"},   # cumulative = 6
        ]
        dd = _compute_drawdown(bets)
        assert dd > 0

    def test_empty_bets_zero_drawdown(self) -> None:
        assert _compute_drawdown([]) == pytest.approx(0.0)

    def test_all_losses_zero_peak_drawdown(self) -> None:
        bets = [
            {"stake": 10.0, "payout": None, "status": "lost"},
            {"stake": 10.0, "payout": None, "status": "lost"},
        ]
        # Peak never exceeds 0, so drawdown = 0
        assert _compute_drawdown(bets) == pytest.approx(0.0)

    def test_pending_bets_ignored(self) -> None:
        bets = [
            {"stake": 10.0, "payout": 18.0, "status": "won"},
            {"stake": 10.0, "payout": None, "status": "pending"},
        ]
        dd = _compute_drawdown(bets)
        assert dd == pytest.approx(0.0)
