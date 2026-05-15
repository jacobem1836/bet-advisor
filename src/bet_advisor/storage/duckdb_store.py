"""
DuckDB analytical store for AFL matches, player stats, team stats, venues, and weather.

This module owns the long-lived analytical/backtest layer. All writes use
parameter binding -- never string interpolation of user data into SQL.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data/analytics.duckdb")

_DDL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    season          INTEGER NOT NULL,
    round           INTEGER,
    date            TIMESTAMP,
    venue           TEXT,
    home_team       TEXT,
    away_team       TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    completed       BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS team_stats (
    match_id            TEXT NOT NULL REFERENCES matches(match_id),
    team                TEXT NOT NULL,
    disposals           INTEGER,
    marks               INTEGER,
    tackles             INTEGER,
    kicks               INTEGER,
    handballs           INTEGER,
    clearances          INTEGER,
    hitouts             INTEGER,
    inside50s           INTEGER,
    contested_poss      INTEGER,
    uncontested_poss    INTEGER,
    PRIMARY KEY (match_id, team)
);

CREATE TABLE IF NOT EXISTS player_stats (
    match_id            TEXT NOT NULL REFERENCES matches(match_id),
    player_id           TEXT NOT NULL,
    player_name         TEXT,
    team                TEXT,
    position            TEXT,
    time_on_ground_pct  DOUBLE,
    disposals           INTEGER,
    kicks               INTEGER,
    handballs           INTEGER,
    marks               INTEGER,
    tackles             INTEGER,
    goals               INTEGER,
    behinds             INTEGER,
    clearances          INTEGER,
    fantasy_points      DOUBLE,
    supercoach_points   DOUBLE,
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS venues (
    venue           TEXT PRIMARY KEY,
    city            TEXT,
    state           TEXT,
    capacity        INTEGER,
    indoor          BOOLEAN DEFAULT FALSE,
    surface         TEXT,
    dimensions_x    DOUBLE,
    dimensions_y    DOUBLE,
    home_teams      JSON
);

CREATE TABLE IF NOT EXISTS weather (
    match_id            TEXT PRIMARY KEY REFERENCES matches(match_id),
    temp_c              DOUBLE,
    wind_kmh            DOUBLE,
    wind_direction_deg  DOUBLE,
    precip_mm           DOUBLE,
    humidity_pct        DOUBLE,
    conditions          TEXT
);
"""


class DuckDBStore:
    """Analytical store backed by DuckDB.

    Designed for read-heavy backtest queries and historical data loads.
    All write methods are idempotent -- safe to call multiple times with
    overlapping data.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._con: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------
    # Lifecycle

    def connect(self) -> "DuckDBStore":
        """Open (or create) the DuckDB file and return self for chaining."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._db_path))
        logger.info("DuckDB connected: %s", self._db_path)
        return self

    def init_schema(self) -> None:
        """Create all tables if they do not already exist."""
        if self._con is None:
            raise RuntimeError("Call connect() before init_schema()")
        self._con.execute(_DDL)
        logger.info("DuckDB schema initialised")

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "DuckDBStore":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._con

    def _upsert_df(self, table: str, df: pd.DataFrame, pk_cols: list[str]) -> int:
        """Generic upsert: insert rows, skip existing PKs.

        DuckDB does not support ON CONFLICT DO UPDATE for multi-col PKs in
        all versions, so we filter out existing keys first then bulk-insert.
        """
        if df.empty:
            return 0

        existing = self.con.execute(
            f"SELECT {', '.join(pk_cols)} FROM {table}"  # noqa: S608
        ).df()

        if not existing.empty:
            merged = df.merge(existing, on=pk_cols, how="left", indicator=True)
            new_rows = df[merged["_merge"] == "left_only"].reset_index(drop=True)
        else:
            new_rows = df

        if new_rows.empty:
            return 0

        self.con.register("_upsert_tmp", new_rows)
        cols = ", ".join(new_rows.columns)
        self.con.execute(
            f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _upsert_tmp"  # noqa: S608
        )
        self.con.unregister("_upsert_tmp")
        logger.debug("Upserted %d rows into %s", len(new_rows), table)
        return len(new_rows)

    # ------------------------------------------------------------------
    # Public write methods

    def upsert_matches(self, df: pd.DataFrame) -> int:
        """Insert new match rows; silently skip rows whose match_id already exists."""
        return self._upsert_df("matches", df, ["match_id"])

    def upsert_team_stats(self, df: pd.DataFrame) -> int:
        """Insert new team_stats rows; skip existing (match_id, team) pairs."""
        return self._upsert_df("team_stats", df, ["match_id", "team"])

    def upsert_player_stats(self, df: pd.DataFrame) -> int:
        """Insert new player_stats rows; skip existing (match_id, player_id) pairs."""
        return self._upsert_df("player_stats", df, ["match_id", "player_id"])

    def upsert_venues(self, df: pd.DataFrame) -> int:
        """Insert new venue rows; skip existing venue names."""
        return self._upsert_df("venues", df, ["venue"])

    def upsert_weather(self, df: pd.DataFrame) -> int:
        """Insert new weather rows; skip existing match_ids."""
        return self._upsert_df("weather", df, ["match_id"])

    # ------------------------------------------------------------------
    # Public read helpers

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute an arbitrary SELECT and return a DataFrame."""
        if params:
            return self.con.execute(sql, params).df()
        return self.con.execute(sql).df()

    def count(self, table: str) -> int:
        """Return the row count for a table."""
        row = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
        return int(row[0]) if row else 0
