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


# ---------------------------------------------------------------------------
# Phase 3: extended schema and new methods
# ---------------------------------------------------------------------------


def _setup_signal(store: SQLiteStore, event_id: str = "EVT1") -> tuple[str, int]:
    """Create an event and a signal; return (event_id, signal_id)."""
    store.upsert_event(
        event_id=event_id,
        sport_key="aussierules_afl",
        sport_title="AFL",
        commence_time="2026-04-01T10:00:00Z",
        home_team="Richmond",
        away_team="Melbourne",
    )
    signal_id = store.insert_signal(
        event_id=event_id,
        market="h2h",
        runner="Richmond",
        model_prob=0.62,
        market_prob_devigged=0.54,
        edge=0.08,
        recommended_stake=1.0,
        model_version="v1",
        created_at="2026-04-01T09:00:00Z",
    )
    return event_id, signal_id


class TestPhase3Schema:
    def test_phase3_columns_present(self, tmp_store: SQLiteStore) -> None:
        """All Phase 3 additional columns must exist on the bets table."""
        info = tmp_store.query("PRAGMA table_info(bets)")
        col_names = {row["name"] for row in info}
        expected = {
            "closing_odds_own",
            "closing_odds_opp",
            "devig_method",
            "recommended_stake_units",
            "actual_stake_units",
            "bankroll_at_placement",
            "expected_value",
            "edge",
            "kelly_fraction",
            "stake_mode",
            "staking_strategy",
        }
        assert expected.issubset(col_names)

    def test_phase3_migration_idempotent(self, tmp_store: SQLiteStore) -> None:
        """Calling connect() again must not error or duplicate columns."""
        tmp_store.connect()
        info = tmp_store.query("PRAGMA table_info(bets)")
        col_names = [row["name"] for row in info]
        # closing_odds_own appears exactly once.
        assert col_names.count("closing_odds_own") == 1


class TestLogSignal:
    def test_log_signal_roundtrip(self, tmp_store: SQLiteStore) -> None:
        tmp_store.upsert_event(
            event_id="EVT2",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time="2026-04-01T10:00:00Z",
            home_team="Geelong",
            away_team="Carlton",
        )
        signal = {
            "event_id": "EVT2",
            "market": "h2h",
            "runner": "Geelong",
            "model_prob": 0.65,
            "market_prob_devigged": 0.55,
            "edge": 0.10,
            "recommended_stake": 1.5,
            "model_version": "disposals-v2",
            "created_at": "2026-04-01T09:30:00Z",
            "rationale": {"key": "model_edge"},
        }
        sig_id = tmp_store.log_signal(signal)
        assert sig_id >= 1

        rows = tmp_store.query("SELECT model_prob, edge FROM signals WHERE id = ?", (sig_id,))
        assert rows[0]["model_prob"] == pytest.approx(0.65)
        assert rows[0]["edge"] == pytest.approx(0.10)


class TestLogBetDict:
    def test_log_bet_with_phase3_fields(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = _setup_signal(tmp_store)
        bet = {
            "signal_id": signal_id,
            "placed_at": "2026-04-01T09:15:00Z",
            "bookmaker": "sportsbet",
            "market": "h2h",
            "runner": "Richmond",
            "price": 1.95,
            "stake": 10.0,
            "expected_value": 0.08,
            "edge": 0.08,
            "kelly_fraction": 0.04,
            "stake_mode": "capped_kelly",
            "staking_strategy": "quarter_kelly_5pct_cap",
            "bankroll_at_placement": 250.0,
            "recommended_stake_units": 0.04,
            "actual_stake_units": 0.04,
            "devig_method": "power",
        }
        bet_id = tmp_store.log_bet_dict(bet)
        assert bet_id >= 1

        rows = tmp_store.query(
            "SELECT expected_value, kelly_fraction, stake_mode FROM bets WHERE id = ?",
            (bet_id,),
        )
        assert rows[0]["expected_value"] == pytest.approx(0.08)
        assert rows[0]["kelly_fraction"] == pytest.approx(0.04)
        assert rows[0]["stake_mode"] == "capped_kelly"

    def test_bet_references_signal_fk(self, tmp_store: SQLiteStore) -> None:
        import sqlite3

        _, signal_id = _setup_signal(tmp_store)
        # Inserting with a non-existent signal_id must fail.
        with pytest.raises(sqlite3.IntegrityError):
            tmp_store.log_bet_dict(
                {
                    "signal_id": 99999,
                    "placed_at": "2026-04-01T09:15:00Z",
                    "bookmaker": "sportsbet",
                    "market": "h2h",
                    "runner": "Richmond",
                    "price": 1.95,
                    "stake": 10.0,
                }
            )


class TestUpdateBetSettlement:
    def test_settlement_updates_clv_and_closing_odds(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = _setup_signal(tmp_store)
        bet_id = tmp_store.log_bet(
            signal_id=signal_id,
            placed_at="2026-04-01T09:30:00Z",
            bookmaker="sportsbet",
            market="h2h",
            runner="Richmond",
            price=1.95,
            stake=10.0,
        )
        tmp_store.update_bet_settlement(
            bet_id=bet_id,
            status="won",
            payout=19.5,
            closing_odds_own=1.88,
            closing_odds_opp=2.02,
            clv_pct=0.025,
            settled_at="2026-04-01T12:30:00Z",
        )
        rows = tmp_store.query(
            "SELECT status, payout, closing_odds_own, closing_odds_opp, clv_pct FROM bets WHERE id=?",
            (bet_id,),
        )
        assert rows[0]["status"] == "won"
        assert rows[0]["payout"] == pytest.approx(19.5)
        assert rows[0]["closing_odds_own"] == pytest.approx(1.88)
        assert rows[0]["closing_odds_opp"] == pytest.approx(2.02)
        assert rows[0]["clv_pct"] == pytest.approx(0.025)

    def test_invalid_status_raises(self, tmp_store: SQLiteStore) -> None:
        _, signal_id = _setup_signal(tmp_store)
        bet_id = tmp_store.log_bet(
            signal_id=signal_id,
            placed_at="2026-04-01T09:30:00Z",
            bookmaker="sportsbet",
            market="h2h",
            runner="Richmond",
            price=1.95,
            stake=10.0,
        )
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.update_bet_settlement(
                bet_id=bet_id,
                status="cancelled",
                payout=None,
                closing_odds_own=None,
                closing_odds_opp=None,
                clv_pct=None,
                settled_at="2026-04-01T12:30:00Z",
            )


class TestComputeCLVSummary:
    def _insert_bets_with_clv(self, store: SQLiteStore, clv_values: list[float]) -> None:
        store.upsert_event(
            event_id="EVTCLV",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time="2026-04-01T10:00:00Z",
            home_team="A",
            away_team="B",
        )
        signal_id = store.insert_signal(
            event_id="EVTCLV",
            market="h2h",
            runner="A",
            model_prob=0.60,
            market_prob_devigged=0.52,
            edge=0.08,
            recommended_stake=1.0,
            model_version="v1",
            created_at="2026-04-01T09:00:00Z",
        )
        for clv in clv_values:
            bet_id = store.log_bet(
                signal_id=signal_id,
                placed_at="2026-04-01T09:30:00Z",
                bookmaker="sportsbet",
                market="h2h",
                runner="A",
                price=1.95,
                stake=10.0,
            )
            store.update_bet_settlement(
                bet_id=bet_id,
                status="won",
                payout=19.5,
                closing_odds_own=1.90,
                closing_odds_opp=2.00,
                clv_pct=clv,
                settled_at="2026-04-01T12:30:00Z",
            )

    def test_summary_keys_present(self, tmp_store: SQLiteStore) -> None:
        self._insert_bets_with_clv(tmp_store, [0.02, -0.01, 0.03, 0.01])
        summary = tmp_store.compute_clv_summary()
        expected_keys = {
            "mean_clv",
            "median_clv",
            "pct_positive",
            "n",
            "wilson_lower",
            "wilson_upper",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_n_matches_inserted(self, tmp_store: SQLiteStore) -> None:
        self._insert_bets_with_clv(tmp_store, [0.02, -0.01, 0.03])
        summary = tmp_store.compute_clv_summary()
        assert summary["n"] == 3

    def test_mean_clv_correct(self, tmp_store: SQLiteStore) -> None:
        values = [0.02, 0.04, -0.01]
        self._insert_bets_with_clv(tmp_store, values)
        summary = tmp_store.compute_clv_summary()
        import numpy as np

        assert summary["mean_clv"] == pytest.approx(np.mean(values), abs=1e-6)
