"""
Model health computation and trigger flag evaluation.

Reads from the model_health SQLite table (idempotently created here) and the
bet log to compute:
  - Last calibration Brier score and ECE
  - Days since last training run
  - Drawdown from peak bankroll
  - Trigger flags per research §6:
      * ECE > 0.02 post-recalibration
      * Drawdown > 25% from peak
      * CLV negative over rolling 100 bets
      * Brier score deteriorates vs two-month prior snapshot
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bet_advisor.storage.duckdb_store import DuckDBStore
    from bet_advisor.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL for model_health table
# ---------------------------------------------------------------------------

_MODEL_HEALTH_DDL = """
CREATE TABLE IF NOT EXISTS model_health (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version   TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    brier           REAL,
    log_loss        REAL,
    ece             REAL,
    mean_clv        REAL,
    drawdown_pct    REAL,
    notes           TEXT
);
"""


def ensure_model_health_table(sqlite_store: SQLiteStore) -> None:
    """Idempotently create the model_health table if it does not exist."""
    sqlite_store.con.executescript(_MODEL_HEALTH_DDL)
    sqlite_store.con.commit()


def record_model_health(
    sqlite_store: SQLiteStore,
    model_version: str,
    brier: float | None = None,
    log_loss: float | None = None,
    ece: float | None = None,
    mean_clv: float | None = None,
    drawdown_pct: float | None = None,
    notes: str | None = None,
) -> int:
    """Insert a model health snapshot and return the new row ID."""
    ensure_model_health_table(sqlite_store)
    now_str = datetime.now(UTC).isoformat(timespec="seconds")
    cur = sqlite_store.con.execute(
        """
        INSERT INTO model_health
            (model_version, captured_at, brier, log_loss, ece, mean_clv, drawdown_pct, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (model_version, now_str, brier, log_loss, ece, mean_clv, drawdown_pct, notes),
    )
    sqlite_store.con.commit()
    return cur.lastrowid or 0


def compute_model_health(
    sqlite_store: SQLiteStore,
    duckdb_store: DuckDBStore | None = None,
) -> dict[str, Any]:
    """Compute a model health summary from stored snapshots and the bet log.

    Parameters
    ----------
    sqlite_store:
        Connected SQLiteStore.  The model_health table is created here if absent.
    duckdb_store:
        Not used in this implementation; reserved for future feature-level health
        checks.  Pass None.

    Returns
    -------
    dict with keys:
        last_model_version   -- version string from most recent snapshot
        last_captured_at     -- ISO timestamp of most recent snapshot
        days_since_snapshot  -- days since last health record (None if no records)
        latest_brier         -- most recent Brier score (None if absent)
        latest_ece           -- most recent ECE (None if absent)
        latest_log_loss      -- most recent log loss (None if absent)
        brier_two_months_ago -- Brier from ~60 days ago (None if absent)
        brier_deteriorated   -- True if brier increased > 0.02 vs two months ago
        drawdown_pct         -- current drawdown from peak bankroll in bet log
        ece_trigger          -- True if latest ECE > 0.02
        drawdown_trigger     -- True if drawdown_pct > 25%
        clv_negative_trigger -- True if mean CLV over rolling 100 bets is negative
        n_bets               -- total bets in the log
        trigger_summary      -- human-readable summary of active triggers
    """
    ensure_model_health_table(sqlite_store)

    # Latest health snapshot
    rows = sqlite_store.query(
        "SELECT * FROM model_health ORDER BY captured_at DESC LIMIT 1"
    )
    latest = rows[0] if rows else {}

    days_since_snapshot: int | None = None
    if latest.get("captured_at"):
        try:
            snap_dt = datetime.fromisoformat(latest["captured_at"])
            if snap_dt.tzinfo is None:
                snap_dt = snap_dt.replace(tzinfo=UTC)
            days_since_snapshot = (datetime.now(UTC) - snap_dt).days
        except ValueError:
            pass

    # Brier two months ago
    two_months_ago = (datetime.now(UTC) - timedelta(days=60)).isoformat(timespec="seconds")
    old_rows = sqlite_store.query(
        "SELECT brier FROM model_health WHERE captured_at <= ? ORDER BY captured_at DESC LIMIT 1",
        (two_months_ago,),
    )
    brier_two_months_ago: float | None = old_rows[0]["brier"] if old_rows else None

    latest_brier: float | None = latest.get("brier")
    brier_deteriorated = False
    if latest_brier is not None and brier_two_months_ago is not None:
        brier_deteriorated = (latest_brier - brier_two_months_ago) > 0.02

    # Bet log analysis
    all_bets = sqlite_store.query("SELECT stake, payout, status, clv_pct, placed_at FROM bets")
    n_bets = len(all_bets)

    # Drawdown from peak bankroll (track cumulative pnl)
    drawdown_pct = _compute_drawdown(all_bets)

    # CLV over rolling 100 bets
    clv_negative_trigger = False
    clv_bets = [b for b in all_bets if b.get("clv_pct") is not None]
    if len(clv_bets) >= 10:
        recent_100 = clv_bets[-100:]
        mean_clv_recent = sum(float(b["clv_pct"]) for b in recent_100) / len(recent_100)
        clv_negative_trigger = mean_clv_recent < 0.0

    # Trigger flags
    latest_ece: float | None = latest.get("ece")
    ece_trigger = (latest_ece is not None) and (latest_ece > 0.02)
    drawdown_trigger = drawdown_pct > 25.0

    # Build summary
    active_triggers: list[str] = []
    if ece_trigger:
        active_triggers.append(f"ECE={latest_ece:.4f} > 0.02 (recalibrate model)")
    if drawdown_trigger:
        active_triggers.append(f"Drawdown={drawdown_pct:.1f}% > 25% (reduce stakes)")
    if clv_negative_trigger:
        active_triggers.append("Mean CLV negative over rolling 100 bets (review model)")
    if brier_deteriorated:
        active_triggers.append(
            f"Brier deteriorated {brier_two_months_ago:.4f} -> {latest_brier:.4f} "
            "> 0.02 over 60 days (retrain)"
        )

    trigger_summary = "; ".join(active_triggers) if active_triggers else "No active triggers"

    return {
        "last_model_version": latest.get("model_version"),
        "last_captured_at": latest.get("captured_at"),
        "days_since_snapshot": days_since_snapshot,
        "latest_brier": latest_brier,
        "latest_ece": latest_ece,
        "latest_log_loss": latest.get("log_loss"),
        "brier_two_months_ago": brier_two_months_ago,
        "brier_deteriorated": brier_deteriorated,
        "drawdown_pct": drawdown_pct,
        "ece_trigger": ece_trigger,
        "drawdown_trigger": drawdown_trigger,
        "clv_negative_trigger": clv_negative_trigger,
        "n_bets": n_bets,
        "trigger_summary": trigger_summary,
    }


def _compute_drawdown(bets: list[dict]) -> float:
    """Compute percentage drawdown from peak bankroll over the bet log.

    Uses cumulative net P&L (payout - stake) to track bankroll trajectory.
    Settled bets contribute real P&L; pending bets use expected value (0 for now).
    Returns drawdown as a positive percentage (0.0 if no bets or no drawdown).
    """
    if not bets:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0

    for bet in bets:
        stake = float(bet.get("stake") or 0)
        payout = bet.get("payout")
        status = bet.get("status", "pending")

        if status == "won" and payout is not None:
            net = float(payout) - stake
        elif status == "lost":
            net = -stake
        else:
            net = 0.0  # pending or void

        cumulative += net
        if cumulative > peak:
            peak = cumulative

        if peak > 0:
            current_drawdown = (peak - cumulative) / peak * 100.0
            if current_drawdown > drawdown:
                drawdown = current_drawdown

    return drawdown
