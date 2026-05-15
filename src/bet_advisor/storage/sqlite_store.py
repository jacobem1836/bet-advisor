"""
SQLite operational store for live odds snapshots, signals, bets, and model versions.

WAL mode is enabled for concurrent reader/writer safety. Foreign keys are on.
All migrations are idempotent CREATE TABLE IF NOT EXISTS statements.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data/operational.db")

_PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
"""

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    sport_key       TEXT NOT NULL,
    sport_title     TEXT,
    commence_time   TEXT NOT NULL,
    home_team       TEXT,
    away_team       TEXT,
    completed       INTEGER DEFAULT 0,
    home_score      REAL,
    away_score      REAL
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL REFERENCES events(event_id),
    bookmaker       TEXT NOT NULL,
    market          TEXT NOT NULL,
    runner          TEXT NOT NULL,
    price           REAL NOT NULL,
    point           REAL,
    captured_at     TEXT NOT NULL,
    commence_time   TEXT,
    source          TEXT NOT NULL DEFAULT 'odds_api'
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_event_id ON odds_snapshots (event_id);
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_captured_at ON odds_snapshots (captured_at);

CREATE TABLE IF NOT EXISTS signals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id             TEXT NOT NULL REFERENCES events(event_id),
    market               TEXT NOT NULL,
    runner               TEXT NOT NULL,
    model_prob           REAL NOT NULL,
    market_prob_devigged REAL NOT NULL,
    edge                 REAL NOT NULL,
    recommended_stake    REAL NOT NULL,
    model_version        TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    rationale            TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS bets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id           INTEGER NOT NULL REFERENCES signals(id),
    placed_at           TEXT NOT NULL,
    bookmaker           TEXT NOT NULL,
    market              TEXT NOT NULL,
    runner              TEXT NOT NULL,
    price               REAL NOT NULL,
    stake               REAL NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'won', 'lost', 'void')),
    payout              REAL,
    closing_price       REAL,
    clv_pct             REAL,
    settled_at          TEXT,
    model_version_hash  TEXT,
    feature_snapshot    TEXT NOT NULL DEFAULT '{}',
    closing_odds_home   REAL,
    closing_odds_away   REAL,
    decision_rationale  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_versions (
    version                 TEXT PRIMARY KEY,
    trained_at              TEXT NOT NULL,
    features_hash           TEXT NOT NULL,
    training_window_start   TEXT NOT NULL,
    training_window_end     TEXT NOT NULL,
    notes                   TEXT
);
"""


class SQLiteStore:
    """Operational store backed by SQLite with WAL mode and foreign key enforcement.

    Use as a context manager or call connect()/close() explicitly.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._con: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle

    def connect(self) -> "SQLiteStore":
        """Open the database, enable WAL + foreign keys, and apply migrations."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_PRAGMAS)
        self._con.executescript(_DDL)
        self._con.commit()
        logger.info("SQLite connected: %s", self._db_path)
        return self

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "SQLiteStore":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._con

    # ------------------------------------------------------------------
    # Events

    def upsert_event(
        self,
        event_id: str,
        sport_key: str,
        sport_title: str,
        commence_time: str,
        home_team: str,
        away_team: str,
        completed: bool = False,
        home_score: float | None = None,
        away_score: float | None = None,
    ) -> None:
        """Insert or replace an event record."""
        self.con.execute(
            """
            INSERT OR REPLACE INTO events
                (event_id, sport_key, sport_title, commence_time,
                 home_team, away_team, completed, home_score, away_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                sport_key,
                sport_title,
                commence_time,
                home_team,
                away_team,
                int(completed),
                home_score,
                away_score,
            ),
        )
        self.con.commit()

    # ------------------------------------------------------------------
    # Odds snapshots

    def insert_snapshot(
        self,
        event_id: str,
        bookmaker: str,
        market: str,
        runner: str,
        price: float,
        captured_at: str,
        point: float | None = None,
        commence_time: str | None = None,
        source: str = "odds_api",
    ) -> int:
        """Insert a single odds snapshot and return the new row ID."""
        cur = self.con.execute(
            """
            INSERT INTO odds_snapshots
                (event_id, bookmaker, market, runner, price, point,
                 captured_at, commence_time, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                bookmaker,
                market,
                runner,
                price,
                point,
                captured_at,
                commence_time,
                source,
            ),
        )
        self.con.commit()
        return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # Signals

    def insert_signal(
        self,
        event_id: str,
        market: str,
        runner: str,
        model_prob: float,
        market_prob_devigged: float,
        edge: float,
        recommended_stake: float,
        model_version: str,
        created_at: str,
        rationale: dict | None = None,
    ) -> int:
        """Insert a new signal and return the new row ID."""
        cur = self.con.execute(
            """
            INSERT INTO signals
                (event_id, market, runner, model_prob, market_prob_devigged,
                 edge, recommended_stake, model_version, created_at, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                market,
                runner,
                model_prob,
                market_prob_devigged,
                edge,
                recommended_stake,
                model_version,
                created_at,
                json.dumps(rationale or {}),
            ),
        )
        self.con.commit()
        return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # Bets

    def log_bet(
        self,
        signal_id: int,
        placed_at: str,
        bookmaker: str,
        market: str,
        runner: str,
        price: float,
        stake: float,
        model_version_hash: str | None = None,
        feature_snapshot: dict | None = None,
        decision_rationale: dict | None = None,
    ) -> int:
        """Record a placed bet and return the new row ID."""
        cur = self.con.execute(
            """
            INSERT INTO bets
                (signal_id, placed_at, bookmaker, market, runner, price, stake,
                 status, model_version_hash, feature_snapshot, decision_rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                signal_id,
                placed_at,
                bookmaker,
                market,
                runner,
                price,
                stake,
                model_version_hash,
                json.dumps(feature_snapshot or {}),
                json.dumps(decision_rationale or {}),
            ),
        )
        self.con.commit()
        return cur.lastrowid or 0

    def settle_bet(
        self,
        bet_id: int,
        status: str,
        payout: float | None,
        closing_price: float | None,
        clv_pct: float | None,
        settled_at: str,
        closing_odds_home: float | None = None,
        closing_odds_away: float | None = None,
    ) -> None:
        """Mark a bet as settled with outcome data."""
        if status not in ("won", "lost", "void"):
            raise ValueError(f"Invalid status: {status!r}")
        self.con.execute(
            """
            UPDATE bets
            SET status=?, payout=?, closing_price=?, clv_pct=?, settled_at=?,
                closing_odds_home=?, closing_odds_away=?
            WHERE id=?
            """,
            (
                status,
                payout,
                closing_price,
                clv_pct,
                settled_at,
                closing_odds_home,
                closing_odds_away,
                bet_id,
            ),
        )
        self.con.commit()

    # ------------------------------------------------------------------
    # Model versions

    def register_model_version(
        self,
        version: str,
        trained_at: str,
        features_hash: str,
        training_window_start: str,
        training_window_end: str,
        notes: str | None = None,
    ) -> None:
        """Record a model version (idempotent -- INSERT OR IGNORE)."""
        self.con.execute(
            """
            INSERT OR IGNORE INTO model_versions
                (version, trained_at, features_hash,
                 training_window_start, training_window_end, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                trained_at,
                features_hash,
                training_window_start,
                training_window_end,
                notes,
            ),
        )
        self.con.commit()

    # ------------------------------------------------------------------
    # General query

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT and return a list of row dicts."""
        rows = self.con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
