"""
Tests for the Scheduler class.

Verifies job registration, dry-run mode, quota enforcement (no live API calls),
and timezone handling. The scheduler is never actually started in these tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_advisor.scheduler import (
    Scheduler,
    _in_final_minute_window,
    _in_pre_bounce_window,
    _quota_exhausted,
    _BRISBANE,
)
from bet_advisor.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_sqlite(tmp_path: Path) -> SQLiteStore:
    db_path = tmp_path / "sched_test.db"
    store = SQLiteStore(db_path)
    store.connect()
    yield store
    store.close()


@pytest.fixture()
def minimal_config(tmp_path: Path) -> dict:
    db_path = tmp_path / "test.db"
    return {
        "odds_api_key": "test-key",
        "odds_api_monthly_budget": 500,
        "odds_api_project_share": 200,
        "odds_api_project_tag": "bet-advisor-test",
        "bankroll": 1000.0,
        "sqlite_path": str(db_path),
        "duckdb_path": str(tmp_path / "test.duckdb"),
        "reports_dir": str(tmp_path / "reports"),
        "stake_mode": "flat",
    }


@pytest.fixture()
def scheduler(minimal_config: dict) -> Scheduler:
    sched = Scheduler(config=minimal_config, dry_run=True)
    sched.register_jobs()
    return sched


# ---------------------------------------------------------------------------
# Job registration tests
# ---------------------------------------------------------------------------


class TestJobRegistration:
    def test_all_expected_jobs_registered(self, scheduler: Scheduler) -> None:
        expected_ids = {
            "refresh_odds_low_freq",
            "refresh_odds_high_freq",
            "refresh_odds_final_min",
            "ingest_results",
            "daily_report",
        }
        registered = set(scheduler.get_registered_job_ids())
        assert expected_ids == registered

    def test_job_count(self, scheduler: Scheduler) -> None:
        assert len(scheduler.get_registered_job_ids()) == 5

    def test_dry_run_does_not_start_scheduler(self, minimal_config: dict) -> None:
        sched = Scheduler(config=minimal_config, dry_run=True)
        sched.register_jobs()
        # Scheduler should not be running after dry-run start
        sched.start()
        assert not sched._scheduler.running


# ---------------------------------------------------------------------------
# Dry-run output test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_print_schedule_outputs_job_names(
        self, scheduler: Scheduler, capsys: pytest.CaptureFixture
    ) -> None:
        scheduler.print_schedule()
        out = capsys.readouterr().out
        assert "refresh_odds_low_freq" in out
        assert "daily_report" in out
        assert "Australia/Brisbane" in out

    def test_print_schedule_shows_project_tag(
        self, scheduler: Scheduler, capsys: pytest.CaptureFixture
    ) -> None:
        scheduler.print_schedule()
        out = capsys.readouterr().out
        assert "bet-advisor-test" in out


# ---------------------------------------------------------------------------
# Quota enforcement tests
# ---------------------------------------------------------------------------


class TestQuotaEnforcement:
    def test_quota_not_exhausted_when_under_limit(self, tmp_sqlite: SQLiteStore) -> None:
        tmp_sqlite.record_quota_usage("bet-advisor", requests_used=50)
        result = _quota_exhausted(tmp_sqlite, "bet-advisor", project_share=200)
        assert result is False

    def test_quota_exhausted_at_limit(self, tmp_sqlite: SQLiteStore) -> None:
        tmp_sqlite.record_quota_usage("bet-advisor", requests_used=200)
        result = _quota_exhausted(tmp_sqlite, "bet-advisor", project_share=200)
        assert result is True

    def test_quota_exhausted_over_limit(self, tmp_sqlite: SQLiteStore) -> None:
        tmp_sqlite.record_quota_usage("bet-advisor", requests_used=300)
        result = _quota_exhausted(tmp_sqlite, "bet-advisor", project_share=200)
        assert result is True

    def test_quota_check_uses_correct_project_tag(self, tmp_sqlite: SQLiteStore) -> None:
        # Record usage for a different project tag
        tmp_sqlite.record_quota_usage("other-project", requests_used=500)
        # bet-advisor should still be under quota
        result = _quota_exhausted(tmp_sqlite, "bet-advisor", project_share=200)
        assert result is False

    def test_job_skips_when_quota_exhausted(
        self, tmp_sqlite: SQLiteStore, minimal_config: dict
    ) -> None:
        """Verify that the low-freq job is a no-op when quota is exhausted."""
        sched = Scheduler(config=minimal_config, dry_run=True)
        sched._sqlite = tmp_sqlite

        # Exhaust the quota
        tmp_sqlite.record_quota_usage("bet-advisor-test", requests_used=500)

        # Patch the OddsAPIClient to verify it is never called
        with patch("bet_advisor.scheduler.OddsAPIClient") as mock_client_cls:
            sched._job_refresh_odds_low_freq()
            mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Timezone handling tests
# ---------------------------------------------------------------------------


class TestTimezoneHandling:
    def test_brisbane_timezone_constant(self) -> None:
        from zoneinfo import ZoneInfo

        assert _BRISBANE == ZoneInfo("Australia/Brisbane")

    def test_pre_bounce_window_false_with_no_matches(self, tmp_sqlite: SQLiteStore) -> None:
        """No matches in DB means no pre-bounce window."""
        result = _in_pre_bounce_window(tmp_sqlite, minutes_before=120)
        assert result is False

    def test_final_minute_window_false_with_no_matches(self, tmp_sqlite: SQLiteStore) -> None:
        result = _in_final_minute_window(tmp_sqlite, minutes_before=30)
        assert result is False

    def test_pre_bounce_window_true_for_upcoming_match(self, tmp_sqlite: SQLiteStore) -> None:
        """Insert an event 60 minutes from now and check the window fires."""
        from datetime import UTC, datetime, timedelta
        from zoneinfo import ZoneInfo

        brisbane = ZoneInfo("Australia/Brisbane")
        # Create an event that commences 60 minutes from now (well within 2h window)
        commence = datetime.now(brisbane) + timedelta(minutes=60)
        commence_str = commence.isoformat()
        today_str = commence.date().isoformat()

        tmp_sqlite.upsert_event(
            event_id="evt-window-test",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time=commence_str,
            home_team="Collingwood",
            away_team="Hawthorn",
            completed=False,
        )
        result = _in_pre_bounce_window(tmp_sqlite, minutes_before=120)
        assert result is True

    def test_pre_bounce_window_false_for_past_match(self, tmp_sqlite: SQLiteStore) -> None:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        brisbane = ZoneInfo("Australia/Brisbane")
        past = datetime.now(brisbane) - timedelta(hours=3)
        tmp_sqlite.upsert_event(
            event_id="evt-past-test",
            sport_key="aussierules_afl",
            sport_title="AFL",
            commence_time=past.isoformat(),
            home_team="Geelong",
            away_team="Richmond",
            completed=False,
        )
        result = _in_pre_bounce_window(tmp_sqlite, minutes_before=120)
        assert result is False
