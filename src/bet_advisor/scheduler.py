"""
APScheduler-based pipeline orchestration.

Wires together odds ingestion, recommendation generation, and daily report
production on a schedule that respects the Odds API quota budget.

Jobs
----
- refresh_odds_low_freq   : hourly (non-match windows)
- refresh_odds_high_freq  : every 15 min (2-hour pre-bounce window)
- refresh_odds_final_min  : every 60 seconds (final 30 min, tracked markets only)
- ingest_results          : daily after games conclude (21:00 AEST)
- daily_report            : every morning at 08:00 AEST

Quota enforcement
-----------------
Every job that touches the Odds API performs a hard skip when MTD usage
>= ODDS_API_PROJECT_SHARE.  The quota check uses SQLite so it survives
process restarts.

Timezone
--------
AFL matches are in Australia/Brisbane (AEST, UTC+10, no DST).
All cron times in this module use ZoneInfo("Australia/Brisbane").
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from bet_advisor.recommend.model_health import compute_model_health, ensure_model_health_table
from bet_advisor.recommend.pnl import compute_pnl_snapshot
from bet_advisor.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_BRISBANE = ZoneInfo("Australia/Brisbane")

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    """Load scheduler configuration from environment variables."""
    return {
        "odds_api_key": os.environ.get("ODDS_API_KEY", ""),
        "odds_api_monthly_budget": int(os.environ.get("ODDS_API_MONTHLY_BUDGET", "500")),
        "odds_api_project_share": int(os.environ.get("ODDS_API_PROJECT_SHARE", "200")),
        "odds_api_project_tag": os.environ.get("ODDS_API_PROJECT_TAG", "bet-advisor"),
        "bankroll": float(os.environ.get("BANKROLL", "1000.0")),
        "sqlite_path": os.environ.get("SQLITE_PATH", "data/operational.db"),
        "duckdb_path": os.environ.get("DUCKDB_PATH", "data/analytics.duckdb"),
        "reports_dir": os.environ.get("REPORTS_DIR", "reports"),
        "stake_mode": os.environ.get("STAKE_MODE", "flat"),
    }


# ---------------------------------------------------------------------------
# Quota helpers
# ---------------------------------------------------------------------------


def _quota_exhausted(sqlite_store: SQLiteStore, project_tag: str, project_share: int) -> bool:
    """Return True if MTD usage has reached the project share budget."""
    year_month = datetime.now(_BRISBANE).strftime("%Y-%m")
    used = sqlite_store.get_month_to_date_usage(project_tag, year_month)
    if used >= project_share:
        logger.warning(
            "Quota exhausted: %d / %d requests used for %s in %s",
            used,
            project_share,
            project_tag,
            year_month,
        )
        return True
    return False


def _record_api_call(sqlite_store: SQLiteStore, project_tag: str, endpoint: str) -> None:
    """Record a single Odds API request to the quota log."""
    sqlite_store.record_quota_usage(project_tag=project_tag, requests_used=1, endpoint=endpoint)


# ---------------------------------------------------------------------------
# AFL schedule helpers
# ---------------------------------------------------------------------------


def _matches_today(sqlite_store: SQLiteStore) -> list[dict[str, Any]]:
    """Return upcoming (not yet completed) events scheduled for today."""
    today = datetime.now(_BRISBANE).date().isoformat()
    rows = sqlite_store.query(
        "SELECT * FROM events WHERE completed = 0 AND commence_time LIKE ?",
        (f"{today}%",),
    )
    return rows


def _active_signal_event_ids(sqlite_store: SQLiteStore) -> list[str]:
    """Return event IDs that have active (pending) signals -- used for final-minute polling."""
    rows = sqlite_store.query(
        """
        SELECT DISTINCT s.event_id FROM signals s
        JOIN bets b ON b.signal_id = s.id
        WHERE b.status = 'pending'
        """
    )
    return [r["event_id"] for r in rows]


def _in_pre_bounce_window(
    sqlite_store: SQLiteStore,
    minutes_before: int = 120,
) -> bool:
    """Return True if any match today is within <minutes_before> minutes of bounce."""
    now = datetime.now(_BRISBANE)
    for match in _matches_today(sqlite_store):
        try:
            commence = datetime.fromisoformat(match["commence_time"])
            if commence.tzinfo is None:
                commence = commence.replace(tzinfo=_BRISBANE)
            diff_minutes = (commence - now).total_seconds() / 60.0
            if 0 < diff_minutes <= minutes_before:
                return True
        except (ValueError, KeyError):
            continue
    return False


def _in_final_minute_window(
    sqlite_store: SQLiteStore,
    minutes_before: int = 30,
) -> bool:
    """Return True if any match today is within the final <minutes_before> minutes window."""
    return _in_pre_bounce_window(sqlite_store, minutes_before=minutes_before)


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------


class Scheduler:
    """APScheduler-based orchestration for the full bet-advisor pipeline.

    Parameters
    ----------
    config:
        Configuration dict. If None, loads from environment variables.
    dry_run:
        If True, registers all jobs but does not start the scheduler.
        Useful for inspecting the schedule without running the pipeline.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._config = config or _load_config()
        self._dry_run = dry_run
        self._sqlite: SQLiteStore | None = None
        self._scheduler = BlockingScheduler(timezone=str(_BRISBANE))
        self._jobs_registered: list[str] = []

    def _get_sqlite(self) -> SQLiteStore:
        """Return a connected SQLiteStore (lazy init)."""
        if self._sqlite is None:
            self._sqlite = SQLiteStore(self._config["sqlite_path"])
            self._sqlite.connect()
            ensure_model_health_table(self._sqlite)
        return self._sqlite

    # ------------------------------------------------------------------
    # Job implementations

    def _job_refresh_odds_low_freq(self) -> None:
        """Hourly odds refresh for non-match windows."""
        sqlite = self._get_sqlite()
        cfg = self._config
        if _quota_exhausted(sqlite, cfg["odds_api_project_tag"], cfg["odds_api_project_share"]):
            return

        # Skip if we are already in a high-frequency window
        if _in_pre_bounce_window(sqlite, minutes_before=120):
            logger.debug("refresh_odds_low_freq: skipping -- in pre-bounce window")
            return

        logger.info("refresh_odds_low_freq: refreshing odds snapshot")
        try:
            from bet_advisor.ingest.odds_api import OddsAPIClient

            client = OddsAPIClient(api_key=cfg["odds_api_key"])
            all_odds = client.fetch_odds()
            _record_api_call(sqlite, cfg["odds_api_project_tag"], "fetch_odds")
            for odds_data in all_odds:
                _store_odds_snapshot(sqlite, odds_data)

        except Exception as exc:
            logger.exception("refresh_odds_low_freq failed: %s", exc)

    def _job_refresh_odds_high_freq(self) -> None:
        """15-minute odds refresh during the 2-hour pre-bounce window."""
        sqlite = self._get_sqlite()
        cfg = self._config
        if _quota_exhausted(sqlite, cfg["odds_api_project_tag"], cfg["odds_api_project_share"]):
            return

        if not _in_pre_bounce_window(sqlite, minutes_before=120):
            logger.debug("refresh_odds_high_freq: not in pre-bounce window, skipping")
            return

        logger.info("refresh_odds_high_freq: refreshing odds (pre-bounce)")
        try:
            from bet_advisor.ingest.odds_api import OddsAPIClient

            client = OddsAPIClient(api_key=cfg["odds_api_key"])
            all_odds = client.fetch_odds()
            _record_api_call(sqlite, cfg["odds_api_project_tag"], "fetch_odds_high")
            today_event_ids = {m["event_id"] for m in _matches_today(sqlite)}
            for odds_data in all_odds:
                if odds_data.event_id in today_event_ids:
                    _store_odds_snapshot(sqlite, odds_data)

        except Exception as exc:
            logger.exception("refresh_odds_high_freq failed: %s", exc)

    def _job_refresh_odds_final_min(self) -> None:
        """60-second odds refresh in the final 30 minutes before bounce.

        Only refreshes markets with active (pending) signals to minimise
        API usage in the most quota-sensitive window.
        """
        sqlite = self._get_sqlite()
        cfg = self._config
        if _quota_exhausted(sqlite, cfg["odds_api_project_tag"], cfg["odds_api_project_share"]):
            return

        if not _in_final_minute_window(sqlite, minutes_before=30):
            return

        active_event_ids = _active_signal_event_ids(sqlite)
        if not active_event_ids:
            logger.debug("refresh_odds_final_min: no active signals, skipping")
            return

        logger.info("refresh_odds_final_min: polling %d tracked events", len(active_event_ids))
        try:
            from bet_advisor.ingest.odds_api import OddsAPIClient

            client = OddsAPIClient(api_key=cfg["odds_api_key"])
            all_odds = client.fetch_odds()
            _record_api_call(sqlite, cfg["odds_api_project_tag"], "fetch_odds_final")
            active_set = set(active_event_ids)
            for odds_data in all_odds:
                if odds_data.event_id in active_set:
                    _store_odds_snapshot(sqlite, odds_data)

        except Exception as exc:
            logger.exception("refresh_odds_final_min failed: %s", exc)

    def _job_ingest_results(self) -> None:
        """Daily result ingestion from Squiggle. Settles pending bets."""
        sqlite = self._get_sqlite()
        logger.info("ingest_results: fetching match results from Squiggle")
        try:
            from bet_advisor.ingest.squiggle import SquiggleClient

            client = SquiggleClient()
            year = datetime.now(_BRISBANE).year
            games = client.fetch_games(year=year)
            for game in games:
                if game.get("complete", 0) == 100:
                    event_id = str(game.get("id", ""))
                    if not event_id:
                        continue
                    sqlite.con.execute(
                        """
                        UPDATE events SET completed=1,
                            home_score=?, away_score=?
                        WHERE event_id=?
                        """,
                        (
                            game.get("hscore"),
                            game.get("ascore"),
                            event_id,
                        ),
                    )
            sqlite.con.commit()
            logger.info("ingest_results: events updated")
        except Exception as exc:
            logger.exception("ingest_results failed: %s", exc)

    def _job_daily_report(self) -> None:
        """08:00 AEST daily report generation."""
        sqlite = self._get_sqlite()
        cfg = self._config

        yesterday = datetime.now(_BRISBANE).date()
        # Report covers today's recommendations
        from bet_advisor.notify import write_markdown_card

        logger.info("daily_report: generating report for %s", yesterday.isoformat())
        try:
            pnl = compute_pnl_snapshot(sqlite, as_of=yesterday, bankroll=cfg["bankroll"])
            health = compute_model_health(sqlite, duckdb_store=None)
            # No live recs for a retrospective report -- pass empty list
            path = write_markdown_card(
                report_date=yesterday,
                recs=[],
                pnl_snapshot=pnl,
                model_health=health,
                sqlite_store=sqlite,
                output_dir=cfg["reports_dir"],
            )
            logger.info("daily_report: written to %s", path)
        except Exception as exc:
            logger.exception("daily_report failed: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle

    def register_jobs(self) -> None:
        """Register all scheduler jobs."""
        # Hourly low-frequency refresh
        self._scheduler.add_job(
            self._job_refresh_odds_low_freq,
            trigger=IntervalTrigger(hours=1, timezone=_BRISBANE),
            id="refresh_odds_low_freq",
            name="Odds refresh (hourly - non-match)",
            max_instances=1,
            coalesce=True,
        )
        self._jobs_registered.append("refresh_odds_low_freq")

        # 15-minute high-frequency refresh
        self._scheduler.add_job(
            self._job_refresh_odds_high_freq,
            trigger=IntervalTrigger(minutes=15, timezone=_BRISBANE),
            id="refresh_odds_high_freq",
            name="Odds refresh (15 min - pre-bounce)",
            max_instances=1,
            coalesce=True,
        )
        self._jobs_registered.append("refresh_odds_high_freq")

        # 60-second final-minute refresh
        self._scheduler.add_job(
            self._job_refresh_odds_final_min,
            trigger=IntervalTrigger(seconds=60, timezone=_BRISBANE),
            id="refresh_odds_final_min",
            name="Odds refresh (60s - final 30 min tracked markets)",
            max_instances=1,
            coalesce=True,
        )
        self._jobs_registered.append("refresh_odds_final_min")

        # Daily result ingestion at 21:00 AEST
        self._scheduler.add_job(
            self._job_ingest_results,
            trigger=CronTrigger(hour=21, minute=0, timezone=_BRISBANE),
            id="ingest_results",
            name="Match result ingestion (daily 21:00 AEST)",
            max_instances=1,
            coalesce=True,
        )
        self._jobs_registered.append("ingest_results")

        # Daily report at 08:00 AEST
        self._scheduler.add_job(
            self._job_daily_report,
            trigger=CronTrigger(hour=8, minute=0, timezone=_BRISBANE),
            id="daily_report",
            name="Daily markdown report (08:00 AEST)",
            max_instances=1,
            coalesce=True,
        )
        self._jobs_registered.append("daily_report")

        logger.info(
            "Registered %d scheduler jobs: %s",
            len(self._jobs_registered),
            ", ".join(self._jobs_registered),
        )

    def print_schedule(self) -> None:
        """Print the planned schedule without starting the scheduler."""
        print("Scheduler configuration (dry-run mode):")
        print("  Timezone: Australia/Brisbane (AEST, UTC+10)")
        print(f"  Project tag: {self._config['odds_api_project_tag']}")
        print(f"  Monthly budget: {self._config['odds_api_monthly_budget']} requests")
        print(f"  Project share: {self._config['odds_api_project_share']} requests")
        print(f"  Bankroll: ${self._config['bankroll']:,.2f}")
        print(f"  Stake mode: {self._config['stake_mode']}")
        print()
        print("Planned jobs:")
        for job in self._scheduler.get_jobs():
            print(f"  [{job.id}] {job.name}")
            print(f"    Trigger: {job.trigger}")

    def start(self) -> None:
        """Register jobs and start the blocking scheduler."""
        self.register_jobs()
        if self._dry_run:
            self.print_schedule()
            logger.info("Dry-run mode: scheduler not started.")
            return
        logger.info("Starting scheduler (blocking). Press Ctrl+C to stop.")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the scheduler and close the SQLite connection."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None

    def get_registered_job_ids(self) -> list[str]:
        """Return the list of registered job IDs (for testing)."""
        return list(self._jobs_registered)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _store_odds_snapshot(sqlite: SQLiteStore, odds_data: Any) -> None:
    """Persist an EventOdds object to SQLite odds_snapshots."""
    if odds_data is None:
        return
    try:
        captured = datetime.now(UTC).isoformat(timespec="seconds")
        for bookmaker in getattr(odds_data, "bookmakers", []):
            for market in getattr(bookmaker, "markets", []):
                for runner in getattr(market, "runners", []):
                    sqlite.insert_snapshot(
                        event_id=odds_data.event_id,
                        bookmaker=bookmaker.key,
                        market=market.key,
                        runner=runner.name,
                        price=runner.price,
                        point=runner.point,
                        captured_at=captured,
                        commence_time=getattr(odds_data, "commence_time", None),
                        source="odds_api",
                    )
    except Exception as exc:
        logger.warning("Failed to store odds snapshot: %s", exc)
