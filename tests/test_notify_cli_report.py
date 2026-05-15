"""
Tests for the MarkdownReport class.

Verifies structure, number formatting, and absence of em dashes.
No HTTP calls are made.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from bet_advisor.notify.cli_report import MarkdownReport
from bet_advisor.recommend.engine import Recommendation
from bet_advisor.storage.sqlite_store import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_sqlite(tmp_path: Path) -> SQLiteStore:
    db_path = tmp_path / "test.db"
    store = SQLiteStore(db_path)
    store.connect()
    yield store
    store.close()


@pytest.fixture()
def sample_recs() -> list[Recommendation]:
    return [
        Recommendation(
            event_id="evt-001",
            market="h2h",
            runner="Collingwood",
            bookmaker="sportsbet",
            decimal_odds=1.85,
            model_prob=0.620,
            model_prob_low=0.590,
            model_prob_high=0.650,
            devigged_market_prob=0.550,
            edge=0.072,
            ev_units=0.072,
            recommended_stake_units=0.01,
            recommended_stake_currency=10.0,
            kelly_fraction=0.10,
            stake_mode="flat",
            confidence_tier="strong",
            rationale={"edge": 0.072, "devig_method": "power"},
            counterarguments=["Edge below 5% -- monitor CLV."],
        ),
        Recommendation(
            event_id="evt-002",
            market="player_disposals",
            runner="Patrick Cripps Over 28.5",
            bookmaker="tab",
            decimal_odds=1.95,
            model_prob=0.580,
            model_prob_low=0.540,
            model_prob_high=0.620,
            devigged_market_prob=0.520,
            edge=0.031,
            ev_units=0.031,
            recommended_stake_units=0.01,
            recommended_stake_currency=10.0,
            kelly_fraction=0.05,
            stake_mode="flat",
            confidence_tier="speculative",
            rationale={"edge": 0.031},
            counterarguments=["Speculative tier -- stake halved."],
        ),
    ]


@pytest.fixture()
def sample_pnl() -> dict:
    return {
        "today_bets": 2,
        "today_units": 20.0,
        "week_bets": 8,
        "week_units": 80.0,
        "month_bets": 25,
        "month_units": 250.0,
        "alltime_bets": 100,
        "alltime_units": 1000.0,
        "alltime_won_units": 1050.0,
        "roi_alltime": 0.05,
        "roi_wilson_lower": 0.01,
        "roi_wilson_upper": 0.09,
        "mean_clv": 0.012,
        "pct_positive_clv": 0.55,
        "n_settled": 80,
        "bankroll": 1000.0,
    }


@pytest.fixture()
def sample_health() -> dict:
    return {
        "last_model_version": "abc12345",
        "last_captured_at": "2026-05-01T08:00:00",
        "days_since_snapshot": 14,
        "latest_brier": 0.2310,
        "latest_ece": 0.0150,
        "latest_log_loss": 0.5500,
        "brier_two_months_ago": 0.2280,
        "brier_deteriorated": False,
        "drawdown_pct": 5.2,
        "ece_trigger": False,
        "drawdown_trigger": False,
        "clv_negative_trigger": False,
        "n_bets": 100,
        "trigger_summary": "No active triggers",
    }


@pytest.fixture()
def reporter(tmp_sqlite: SQLiteStore, tmp_path: Path) -> MarkdownReport:
    return MarkdownReport(sqlite_store=tmp_sqlite, output_dir=tmp_path / "reports")


# ---------------------------------------------------------------------------
# render_daily structure tests
# ---------------------------------------------------------------------------


class TestRenderDailyStructure:
    def test_contains_header(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "# Bet Advisor" in md
        assert "2026-05-15" in md

    def test_contains_top_recommendations_section(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## Top Recommendations" in md

    def test_contains_bankroll_section(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## Bankroll & P&L Snapshot" in md

    def test_contains_model_health_section(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## Model Health" in md

    def test_contains_risk_flags_section(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## Risk Flags" in md

    def test_contains_quota_section(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## API Quota" in md

    def test_recommendation_detail_section_present(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "## Recommendation Detail" in md

    def test_runner_names_in_report(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "Collingwood" in md
        assert "Patrick Cripps" in md

    def test_confidence_tiers_in_report(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "strong" in md
        assert "speculative" in md

    def test_no_recs_message_when_empty(
        self,
        reporter: MarkdownReport,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), [], sample_pnl, sample_health)
        assert "No recommendations" in md


# ---------------------------------------------------------------------------
# Number formatting tests
# ---------------------------------------------------------------------------


class TestNumberFormatting:
    def test_odds_formatted_to_two_dp(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        # Collingwood odds are 1.85 -- should appear as "1.85"
        assert "1.85" in md

    def test_edge_formatted_to_three_dp(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        # Edge 0.072 should appear as "0.072"
        assert "0.072" in md

    def test_brier_formatted_to_four_dp(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "0.2310" in md

    def test_roi_formatted_as_percentage(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        # ROI is 0.05 -- should appear as "5.0%"
        assert "5.0%" in md


# ---------------------------------------------------------------------------
# Em dash policy test
# ---------------------------------------------------------------------------


class TestNoEmDashes:
    def test_no_em_dashes_in_rendered_report(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        assert "—" not in md, "Report must not contain em dashes (U+2014)"

    def test_no_em_dashes_in_empty_report(
        self,
        reporter: MarkdownReport,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        md = reporter.render_daily(date(2026, 5, 15), [], sample_pnl, sample_health)
        assert "—" not in md, "Empty report must not contain em dashes (U+2014)"


# ---------------------------------------------------------------------------
# write_daily file creation tests
# ---------------------------------------------------------------------------


class TestWriteDaily:
    def test_writes_file_at_correct_path(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
        tmp_path: Path,
    ) -> None:
        report_date = date(2026, 5, 15)
        path = reporter.write_daily(report_date, sample_recs, sample_pnl, sample_health)
        assert path.exists()
        assert path.name == "2026-05-15.md"

    def test_written_file_contains_header(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        path = reporter.write_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        content = path.read_text(encoding="utf-8")
        assert "# Bet Advisor" in content

    def test_written_file_is_utf8(
        self,
        reporter: MarkdownReport,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
    ) -> None:
        path = reporter.write_daily(date(2026, 5, 15), sample_recs, sample_pnl, sample_health)
        # Should not raise
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_write_daily_creates_output_dir(
        self, tmp_sqlite: SQLiteStore, tmp_path: Path, sample_pnl: dict, sample_health: dict
    ) -> None:
        new_dir = tmp_path / "new-reports-dir"
        assert not new_dir.exists()
        reporter = MarkdownReport(sqlite_store=tmp_sqlite, output_dir=new_dir)
        reporter.write_daily(date(2026, 5, 15), [], sample_pnl, sample_health)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# write_markdown_card convenience function
# ---------------------------------------------------------------------------


class TestWriteMarkdownCard:
    def test_convenience_function(
        self,
        tmp_sqlite: SQLiteStore,
        sample_recs: list,
        sample_pnl: dict,
        sample_health: dict,
        tmp_path: Path,
    ) -> None:
        from bet_advisor.notify import write_markdown_card

        path = write_markdown_card(
            report_date=date(2026, 5, 15),
            recs=sample_recs,
            pnl_snapshot=sample_pnl,
            model_health=sample_health,
            sqlite_store=tmp_sqlite,
            output_dir=tmp_path / "cards",
        )
        assert path.exists()
        assert path.suffix == ".md"
