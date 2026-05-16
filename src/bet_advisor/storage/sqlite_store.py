"""
SQLite operational store for live odds snapshots, signals, bets, and model versions.

WAL mode is enabled for concurrent reader/writer safety. Foreign keys are on.
All migrations are idempotent CREATE TABLE IF NOT EXISTS statements.

Phase 3 additions:

- Extended bets table: closing_odds_own/opp, devig_method, recommended/actual stake
  units, bankroll_at_placement, expected_value, edge, kelly_fraction, stake_mode,
  staking_strategy fields (idempotent ALTER TABLE migration at connect time).
- New methods: log_signal, log_bet (dict-based), update_bet_settlement,
  compute_clv_summary.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, date, datetime
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

CREATE TABLE IF NOT EXISTS quota_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_tag TEXT NOT NULL,
    date TEXT NOT NULL,
    requests_used INTEGER NOT NULL DEFAULT 0,
    endpoint TEXT,
    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quota_usage_project_date ON quota_usage(project_tag, date);
"""

# Idempotent ALTER TABLE migrations for Phase 3 bet-log extensions.
# Each entry is the column name and its DDL fragment.  connect() will check
# PRAGMA table_info(bets) and only add columns that are absent.
_BETS_PHASE3_COLUMNS: list[tuple[str, str]] = [
    ("closing_odds_own", "REAL"),
    ("closing_odds_opp", "REAL"),
    ("devig_method", "TEXT"),
    ("recommended_stake_units", "REAL"),
    ("actual_stake_units", "REAL"),
    ("bankroll_at_placement", "REAL"),
    ("expected_value", "REAL"),
    ("edge", "REAL"),
    ("kelly_fraction", "REAL"),
    ("stake_mode", "TEXT"),
    ("staking_strategy", "TEXT"),
]

# Phase 5.5 columns: CLV reference source tracking.
_BETS_PHASE55_COLUMNS: list[tuple[str, str]] = [
    ("clv_reference_source", "TEXT"),
    ("clv_reference_books_used", "TEXT"),
]


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
        self._apply_phase3_migrations()
        self._apply_phase55_migrations()
        self._con.commit()
        logger.info("SQLite connected: %s", self._db_path)
        return self

    def _apply_phase3_migrations(self) -> None:
        """Idempotently add Phase 3 columns to the bets table."""
        assert self._con is not None
        existing = {row[1] for row in self._con.execute("PRAGMA table_info(bets)").fetchall()}
        for col_name, col_type in _BETS_PHASE3_COLUMNS:
            if col_name not in existing:
                self._con.execute(f"ALTER TABLE bets ADD COLUMN {col_name} {col_type}")
                logger.debug("Added column bets.%s (%s)", col_name, col_type)

    def _apply_phase55_migrations(self) -> None:
        """Idempotently add Phase 5.5 CLV reference columns to the bets table."""
        assert self._con is not None
        existing = {row[1] for row in self._con.execute("PRAGMA table_info(bets)").fetchall()}
        for col_name, col_type in _BETS_PHASE55_COLUMNS:
            if col_name not in existing:
                self._con.execute(f"ALTER TABLE bets ADD COLUMN {col_name} {col_type}")
                logger.debug("Added column bets.%s (%s)", col_name, col_type)

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
    # Phase 3: dict-based signal / bet helpers

    def log_signal(self, signal: dict[str, Any]) -> int:
        """Insert a signal from a dict and return the new row ID.

        Required keys: ``event_id``, ``market``, ``runner``, ``model_prob``,
        ``market_prob_devigged``, ``edge``, ``recommended_stake``,
        ``model_version``, ``created_at``.

        Optional keys: ``rationale`` (dict, defaults to ``{}``).
        """
        return self.insert_signal(
            event_id=signal["event_id"],
            market=signal["market"],
            runner=signal["runner"],
            model_prob=float(signal["model_prob"]),
            market_prob_devigged=float(signal["market_prob_devigged"]),
            edge=float(signal["edge"]),
            recommended_stake=float(signal["recommended_stake"]),
            model_version=signal["model_version"],
            created_at=signal["created_at"],
            rationale=signal.get("rationale"),
        )

    def log_bet_dict(self, bet: dict[str, Any]) -> int:
        """Insert a bet from a dict (Phase 3 extended schema) and return the row ID.

        Required keys: ``signal_id``, ``placed_at``, ``bookmaker``, ``market``,
        ``runner``, ``price``, ``stake``.

        Optional Phase 3 keys (stored when present):
        ``model_version_hash``, ``feature_snapshot``, ``decision_rationale``,
        ``closing_odds_own``, ``closing_odds_opp``, ``devig_method``,
        ``recommended_stake_units``, ``actual_stake_units``,
        ``bankroll_at_placement``, ``expected_value``, ``edge``,
        ``kelly_fraction``, ``stake_mode``, ``staking_strategy``.
        """
        # Insert core fields first.
        bet_id = self.log_bet(
            signal_id=int(bet["signal_id"]),
            placed_at=bet["placed_at"],
            bookmaker=bet["bookmaker"],
            market=bet["market"],
            runner=bet["runner"],
            price=float(bet["price"]),
            stake=float(bet["stake"]),
            model_version_hash=bet.get("model_version_hash"),
            feature_snapshot=bet.get("feature_snapshot"),
            decision_rationale=bet.get("decision_rationale"),
        )

        # Patch in Phase 3 extended fields if present.
        phase3_keys = [col for col, _ in _BETS_PHASE3_COLUMNS]
        patch: dict[str, Any] = {k: bet[k] for k in phase3_keys if k in bet}
        if patch:
            set_clause = ", ".join(f"{k} = ?" for k in patch)
            self.con.execute(
                f"UPDATE bets SET {set_clause} WHERE id = ?",
                (*patch.values(), bet_id),
            )
            self.con.commit()

        return bet_id

    def update_bet_settlement(
        self,
        bet_id: int,
        status: str,
        payout: float | None,
        closing_odds_own: float | None,
        closing_odds_opp: float | None,
        clv_pct: float | None,
        settled_at: str,
        clv_reference_source: str | None = None,
        clv_reference_books_used: list[str] | None = None,
    ) -> None:
        """Update a bet with post-settlement data including CLV and closing odds.

        Parameters
        ----------
        bet_id:
            Row ID of the bet to update.
        status:
            One of ``"won"``, ``"lost"``, ``"void"``.
        payout:
            Gross payout in stake units (``None`` for void).
        closing_odds_own:
            Closing decimal odds for the betted side.
        closing_odds_opp:
            Closing decimal odds for the opposing side.
        clv_pct:
            Computed CLV as a decimal (e.g. 0.025 for +2.5pp).
        settled_at:
            ISO 8601 timestamp of settlement.
        clv_reference_source:
            Source identifier for the CLV reference (e.g.
            ``"consensus:sportsbet,tab,ladbrokes"`` or ``"betfair_delayed"``).
            Stored for audit and reporting. Optional.
        clv_reference_books_used:
            List of bookmaker keys that contributed to the CLV reference.
            Serialised as JSON. Optional.
        """
        if status not in ("won", "lost", "void"):
            raise ValueError(f"Invalid status: {status!r}")
        books_json = json.dumps(clv_reference_books_used) if clv_reference_books_used is not None else None
        self.con.execute(
            """
            UPDATE bets
            SET status=?, payout=?, closing_odds_own=?, closing_odds_opp=?,
                clv_pct=?, settled_at=?,
                clv_reference_source=?, clv_reference_books_used=?
            WHERE id=?
            """,
            (
                status,
                payout,
                closing_odds_own,
                closing_odds_opp,
                clv_pct,
                settled_at,
                clv_reference_source,
                books_json,
                bet_id,
            ),
        )
        self.con.commit()

    def compute_clv_summary(
        self,
        since: date | None = None,
    ) -> dict[str, float | int]:
        """Compute aggregate CLV statistics over settled bets.

        Fetches all bets with non-null ``clv_pct`` (optionally filtered by
        ``placed_at`` date) and delegates to ``eval.clv.aggregate_clv``.

        Parameters
        ----------
        since:
            Optional lower bound on ``placed_at`` (inclusive).  If ``None``,
            all settled bets with a CLV value are included.

        Returns
        -------
        dict
            Output of :func:`bet_advisor.eval.clv.aggregate_clv`:
            ``mean_clv``, ``median_clv``, ``pct_positive``, ``n``,
            ``wilson_lower``, ``wilson_upper``.

        Raises
        ------
        ValueError
            If no settled bets with CLV data are found.
        """
        import pandas as pd

        from bet_advisor.eval.clv import aggregate_clv

        if since is not None:
            rows = self.query(
                "SELECT clv_pct FROM bets WHERE clv_pct IS NOT NULL AND placed_at >= ?",
                (since.isoformat(),),
            )
        else:
            rows = self.query("SELECT clv_pct FROM bets WHERE clv_pct IS NOT NULL")

        df = pd.DataFrame(rows)
        return aggregate_clv(df)

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
    # Quota usage

    def record_quota_usage(
        self,
        project_tag: str,
        requests_used: int,
        endpoint: str | None = None,
    ) -> None:
        """Record Odds API quota consumption for budget tracking."""
        date = datetime.now(UTC).date().isoformat()
        self.con.execute(
            """
            INSERT INTO quota_usage
                (project_tag, date, requests_used, endpoint, recorded_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (project_tag, date, requests_used, endpoint),
        )
        self.con.commit()

    def get_month_to_date_usage(self, project_tag: str, year_month: str) -> int:
        """Return total requests_used for project_tag where date LIKE year_month||'%'."""
        row = self.con.execute(
            """
            SELECT COALESCE(SUM(requests_used), 0) as total
            FROM quota_usage
            WHERE project_tag = ? AND date LIKE ?
            """,
            (project_tag, f"{year_month}%"),
        ).fetchone()
        return row["total"] if row else 0

    # ------------------------------------------------------------------
    # General query

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a SELECT and return a list of row dicts."""
        rows = self.con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
