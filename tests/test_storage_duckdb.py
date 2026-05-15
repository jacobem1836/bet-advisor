"""Tests for DuckDBStore -- schema init, upsert idempotency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bet_advisor.storage.duckdb_store import DuckDBStore


@pytest.fixture
def tmp_db(tmp_path: Path) -> DuckDBStore:
    store = DuckDBStore(tmp_path / "test.duckdb")
    store.connect()
    store.init_schema()
    yield store
    store.close()


class TestSchemaInit:
    def test_init_schema_creates_tables(self, tmp_db: DuckDBStore) -> None:
        tables = tmp_db.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        )
        names = set(tables["table_name"].tolist())
        expected = {"matches", "team_stats", "player_stats", "venues", "weather"}
        assert expected.issubset(names)

    def test_init_schema_is_idempotent(self, tmp_db: DuckDBStore) -> None:
        # Calling init_schema twice must not raise
        tmp_db.init_schema()
        tmp_db.init_schema()
        assert tmp_db.count("matches") == 0

    def test_count_empty_table(self, tmp_db: DuckDBStore) -> None:
        assert tmp_db.count("matches") == 0
        assert tmp_db.count("player_stats") == 0


class TestUpsertMatches:
    def _sample_matches(self, n: int = 3) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "match_id": f"M{i}",
                    "season": 2024,
                    "round": i,
                    "date": "2024-03-01",
                    "venue": "MCG",
                    "home_team": "Richmond",
                    "away_team": "Melbourne",
                    "home_score": 90,
                    "away_score": 80,
                    "completed": True,
                }
                for i in range(1, n + 1)
            ]
        )

    def test_upsert_inserts_new_rows(self, tmp_db: DuckDBStore) -> None:
        df = self._sample_matches(3)
        inserted = tmp_db.upsert_matches(df)
        assert inserted == 3
        assert tmp_db.count("matches") == 3

    def test_upsert_skips_duplicates(self, tmp_db: DuckDBStore) -> None:
        df = self._sample_matches(3)
        tmp_db.upsert_matches(df)
        # Second call with the same data must insert 0 new rows
        inserted_again = tmp_db.upsert_matches(df)
        assert inserted_again == 0
        assert tmp_db.count("matches") == 3

    def test_upsert_partial_overlap(self, tmp_db: DuckDBStore) -> None:
        df_first = self._sample_matches(2)
        tmp_db.upsert_matches(df_first)

        # 3 rows, first 2 are existing, 3rd is new
        df_second = self._sample_matches(3)
        inserted = tmp_db.upsert_matches(df_second)
        assert inserted == 1
        assert tmp_db.count("matches") == 3

    def test_upsert_empty_df_is_noop(self, tmp_db: DuckDBStore) -> None:
        inserted = tmp_db.upsert_matches(pd.DataFrame())
        assert inserted == 0

    def test_query_returns_correct_data(self, tmp_db: DuckDBStore) -> None:
        df = self._sample_matches(1)
        tmp_db.upsert_matches(df)
        result = tmp_db.query("SELECT match_id, home_team FROM matches WHERE match_id = ?", ["M1"])
        assert len(result) == 1
        assert result.iloc[0]["home_team"] == "Richmond"


class TestUpsertTeamStats:
    def test_upsert_team_stats(self, tmp_db: DuckDBStore) -> None:
        # Must have a matching match first (FK)
        matches_df = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "season": 2024,
                    "round": 1,
                    "date": "2024-03-01",
                    "venue": "MCG",
                    "home_team": "Richmond",
                    "away_team": "Melbourne",
                    "home_score": 90,
                    "away_score": 80,
                    "completed": True,
                }
            ]
        )
        tmp_db.upsert_matches(matches_df)

        stats_df = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "team": "Richmond",
                    "disposals": 320,
                    "marks": 80,
                    "tackles": 60,
                    "kicks": 180,
                    "handballs": 140,
                    "clearances": 40,
                    "hitouts": 35,
                    "inside50s": 55,
                    "contested_poss": 120,
                    "uncontested_poss": 200,
                }
            ]
        )
        inserted = tmp_db.upsert_team_stats(stats_df)
        assert inserted == 1
        assert tmp_db.count("team_stats") == 1

        # Second upsert same data -- should insert 0
        inserted_again = tmp_db.upsert_team_stats(stats_df)
        assert inserted_again == 0


class TestUpsertVenues:
    def test_upsert_venues(self, tmp_db: DuckDBStore) -> None:
        df = pd.DataFrame(
            [
                {
                    "venue": "MCG",
                    "city": "Melbourne",
                    "state": "VIC",
                    "capacity": 100024,
                    "indoor": False,
                    "surface": "grass",
                    "dimensions_x": None,
                    "dimensions_y": None,
                    "home_teams": '["Richmond"]',
                }
            ]
        )
        inserted = tmp_db.upsert_venues(df)
        assert inserted == 1
        # Idempotent
        inserted_again = tmp_db.upsert_venues(df)
        assert inserted_again == 0
