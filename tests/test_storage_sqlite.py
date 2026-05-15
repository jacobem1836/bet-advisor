"""Tests for SQLiteStore -- schema init, FK enforcement, WAL mode, bet log roundtrip."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bet_advisor.storage.sqlite_store import SQLiteStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> SQLiteStore:
    store = SQLiteStore(tmp_path / "test.db")
    store.connect()
    yield store
    store.close()


class TestSchemaInit:
    def test_tables_created(self, tmp_store: SQLiteStore) -> None:
        rows = tmp_store.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        names = {r["name"] for r in rows}
        expected = {"bets", "events", "model_versions", "odds_snapshots", "signals"}
        assert expected.issubset(names)

    def test_wal_mode(self, tmp_store: SQLiteStore) -> None:
        rows = tmp_store.query("PRAGMA journal_mode")
        assert rows[0]["journal_mode"] == "wal"

    def test_foreign_keys_on(self, tmp_store: SQLiteStore) -> None:
        rows = tmp_store.query("PRAGMA foreign_keys")
        assert rows[0]["foreign_keys"] == 1

    def test_init_is_idempotent(self, tmp_store: SQLiteStore) -> None:
        # connect() is called again -- must not raise or duplicate tables
        tmp_store.connect()
        rows = tmp_store.query("SELECT COUNT(*) AS cnt FROM sqlite_master WHERE type='table'")
        assert rows[0]["cnt"] >= 5


class TestEventUpsert:
    def test_insert_event(self, tmp_store: SQLiteStore) -> None:
        tmp_store.upsert_event(
            event_id="EVT1",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time="2026-04-01T10:00:00Z",
            home_team="Richmond",
            away_team="Melbourne",
        )
        rows = tmp_store.query("SELECT event_id FROM events")
        assert len(rows) == 1
        assert rows[0]["event_id"] == "EVT1"

    def test_replace_event(self, tmp_store: SQLiteStore) -> None:
        for score in [80, 90]:
            tmp_store.upsert_event(
                event_id="EVT1",
                sport_key="aussierules_afl",
                sport_title="AFL",
                commence_time="2026-04-01T10:00:00Z",
                home_team="Richmond",
                away_team="Melbourne",
                completed=True,
                home_score=float(score),
            )
        rows = tmp_store.query("SELECT home_score FROM events WHERE event_id = 'EVT1'")
        assert rows[0]["home_score"] == 90.0


class TestOddsSnapshots:
    def _setup_event(self, store: SQLiteStore) -> None:
        store.upsert_event(
            event_id="EVT1",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time="2026-04-01T10:00:00Z",
            home_team="Richmond",
            away_team="Melbourne",
        )

    def test_insert_snapshot_returns_id(self, tmp_store: SQLiteStore) -> None:
        self._setup_event(tmp_store)
        row_id = tmp_store.insert_snapshot(
            event_id="EVT1",
            bookmaker="sportsbet",
            market="h2h",
            runner="Richmond",
            price=1.85,
            captured_at="2026-04-01T08:00:00Z",
        )
        assert row_id >= 1

    def test_multiple_snapshots(self, tmp_store: SQLiteStore) -> None:
        self._setup_event(tmp_store)
        for i, price in enumerate([1.85, 1.80, 1.78], start=1):
            tmp_store.insert_snapshot(
                event_id="EVT1",
                bookmaker="sportsbet",
                market="h2h",
                runner="Richmond",
                price=price,
                captured_at=f"2026-04-01T0{i + 7}:00:00Z",
            )
        rows = tmp_store.query("SELECT COUNT(*) AS cnt FROM odds_snapshots")
        assert rows[0]["cnt"] == 3

    def test_foreign_key_enforced(self, tmp_store: SQLiteStore) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            tmp_store.insert_snapshot(
                event_id="NONEXISTENT",
                bookmaker="sportsbet",
                market="h2h",
                runner="Richmond",
                price=1.85,
                captured_at="2026-04-01T08:00:00Z",
            )


class TestBetLogRoundtrip:
    def _setup(self, store: SQLiteStore) -> tuple[str, int]:
        """Create an event + signal and return (event_id, signal_id)."""
        store.upsert_event(
            event_id="EVT1",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time="2026-04-01T10:00:00Z",
            home_team="Richmond",
            away_team="Melbourne",
        )
        signal_id = store.insert_signal(
            event_id="EVT1",
            market="player_disposals",
            runner="Over 24.5",
            model_prob=0.60,
            market_prob_devigged=0.52,
            edge=0.08,
            recommended_stake=1.0,
            model_version="disposals-v1",
            created_at="2026-04-01T09:00:00Z",
            rationale={"feature": "ewm_disposals", "value": 26.3},
        )
        return "EVT1", signal_id

    def test_bet_log_roundtrip(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = self._setup(tmp_store)

        bet_id = tmp_store.log_bet(
            signal_id=signal_id,
            placed_at="2026-04-01T09:30:00Z",
            bookmaker="sportsbet",
            market="player_disposals",
            runner="Over 24.5",
            price=1.95,
            stake=1.0,
            feature_snapshot={"ewm_disposals": 26.3},
        )
        assert bet_id >= 1

        rows = tmp_store.query("SELECT status, price, stake FROM bets WHERE id = ?", (bet_id,))
        assert rows[0]["status"] == "pending"
        assert rows[0]["price"] == 1.95
        assert rows[0]["stake"] == 1.0

    def test_settle_bet(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = self._setup(tmp_store)
        bet_id = tmp_store.log_bet(
            signal_id=signal_id,
            placed_at="2026-04-01T09:30:00Z",
            bookmaker="sportsbet",
            market="h2h",
            runner="Richmond",
            price=1.85,
            stake=1.0,
        )
        tmp_store.settle_bet(
            bet_id=bet_id,
            status="won",
            payout=1.85,
            closing_price=1.82,
            clv_pct=1.65,
            settled_at="2026-04-01T12:00:00Z",
        )
        rows = tmp_store.query("SELECT status, payout, clv_pct FROM bets WHERE id = ?", (bet_id,))
        assert rows[0]["status"] == "won"
        assert rows[0]["payout"] == pytest.approx(1.85)
        assert rows[0]["clv_pct"] == pytest.approx(1.65)

    def test_invalid_status_raises(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = self._setup(tmp_store)
        bet_id = tmp_store.log_bet(
            signal_id=signal_id,
            placed_at="2026-04-01T09:30:00Z",
            bookmaker="sportsbet",
            market="h2h",
            runner="Richmond",
            price=1.85,
            stake=1.0,
        )
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.settle_bet(
                bet_id=bet_id,
                status="cancelled",
                payout=None,
                closing_price=None,
                clv_pct=None,
                settled_at="2026-04-01T12:00:00Z",
            )


class TestModelVersions:
    def test_register_model_version(self, tmp_store: SQLiteStore) -> None:
        tmp_store.register_model_version(
            version="disposals-v1",
            trained_at="2026-04-01T00:00:00Z",
            features_hash="abc123",
            training_window_start="2019-01-01",
            training_window_end="2025-12-31",
            notes="Initial disposals model",
        )
        rows = tmp_store.query("SELECT version FROM model_versions")
        assert any(r["version"] == "disposals-v1" for r in rows)

    def test_register_is_idempotent(self, tmp_store: SQLiteStore) -> None:
        for _ in range(3):
            tmp_store.register_model_version(
                version="disposals-v1",
                trained_at="2026-04-01T00:00:00Z",
                features_hash="abc123",
                training_window_start="2019-01-01",
                training_window_end="2025-12-31",
            )
        rows = tmp_store.query("SELECT COUNT(*) AS cnt FROM model_versions")
        assert rows[0]["cnt"] == 1


class TestQuotaUsage:
    def test_record_quota_usage_inserts_row(self, tmp_store: SQLiteStore) -> None:
        tmp_store.record_quota_usage(
            project_tag="bet-advisor",
            requests_used=42,
            endpoint="sports",
        )
        rows = tmp_store.query("SELECT requests_used, endpoint FROM quota_usage")
        assert len(rows) == 1
        assert rows[0]["requests_used"] == 42
        assert rows[0]["endpoint"] == "sports"

    def test_get_month_to_date_usage_sums_correctly(self, tmp_store: SQLiteStore) -> None:
        # Insert 3 rows in 2026-05
        for i in range(1, 4):
            tmp_store.con.execute(
                """
                INSERT INTO quota_usage
                    (project_tag, date, requests_used, endpoint, recorded_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                ("bet-advisor", f"2026-05-{i:02d}", 100, "sports"),
            )
        tmp_store.con.commit()

        # Insert 1 row in 2026-06 (different month)
        tmp_store.con.execute(
            """
            INSERT INTO quota_usage
                (project_tag, date, requests_used, endpoint, recorded_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            ("bet-advisor", "2026-06-01", 100, "sports"),
        )
        tmp_store.con.commit()

        # Assert 2026-05 total is 300 (3 rows * 100)
        total = tmp_store.get_month_to_date_usage("bet-advisor", "2026-05")
        assert total == 300

        # Assert 2026-06 total is 100 (1 row * 100)
        total = tmp_store.get_month_to_date_usage("bet-advisor", "2026-06")
        assert total == 100
