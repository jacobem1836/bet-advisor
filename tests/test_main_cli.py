"""
Tests for the CLI argparse interface.

Verifies that each sub-command routes to the correct handler.
No live storage, no HTTP calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_advisor.main import build_parser, cli, cmd_quota, cmd_health


# ---------------------------------------------------------------------------
# Parser structure tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_schedule_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["schedule"])
        assert args.command == "schedule"
        assert args.dry_run is False

    def test_schedule_dry_run_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["schedule", "--dry-run"])
        assert args.dry_run is True

    def test_recommend_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recommend", "--round", "5"])
        assert args.command == "recommend"
        assert args.round == 5
        assert args.persist is False

    def test_recommend_persist_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recommend", "--round", "5", "--persist"])
        assert args.persist is True

    def test_recommend_allow_untrained_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["recommend", "--round", "5", "--allow-untrained"])
        assert args.allow_untrained is True

    def test_report_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["report", "--date", "2026-05-15"])
        assert args.command == "report"
        assert args.date == "2026-05-15"

    def test_backtest_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["backtest"])
        assert args.command == "backtest"
        assert args.config is None

    def test_quota_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["quota"])
        assert args.command == "quota"

    def test_health_command_exists(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_no_subcommand_raises_error(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Route dispatching tests (handlers are mocked)
# ---------------------------------------------------------------------------


class TestCommandRouting:
    def _run_cli(self, argv: list[str]) -> int:
        import sys

        with patch.object(sys, "argv", ["bet-advisor"] + argv):
            with pytest.raises(SystemExit) as exc_info:
                cli()
        return exc_info.value.code

    def test_schedule_routes_to_scheduler(self) -> None:
        with patch("bet_advisor.main.cmd_schedule", return_value=0) as mock_cmd:
            self._run_cli(["schedule", "--dry-run"])
            mock_cmd.assert_called_once()

    def test_recommend_routes_to_handler(self) -> None:
        with patch("bet_advisor.main.cmd_recommend", return_value=0) as mock_cmd:
            self._run_cli(["recommend", "--round", "5"])
            mock_cmd.assert_called_once()

    def test_report_routes_to_handler(self) -> None:
        with patch("bet_advisor.main.cmd_report", return_value=0) as mock_cmd:
            self._run_cli(["report", "--date", "2026-05-15"])
            mock_cmd.assert_called_once()

    def test_quota_routes_to_handler(self) -> None:
        with patch("bet_advisor.main.cmd_quota", return_value=0) as mock_cmd:
            self._run_cli(["quota"])
            mock_cmd.assert_called_once()

    def test_health_routes_to_handler(self) -> None:
        with patch("bet_advisor.main.cmd_health", return_value=0) as mock_cmd:
            self._run_cli(["health"])
            mock_cmd.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_quota integration test (real SQLite, no HTTP)
# ---------------------------------------------------------------------------


class TestCmdQuota:
    def test_quota_prints_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from bet_advisor.storage.sqlite_store import SQLiteStore

        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        store.connect()
        store.record_quota_usage("bet-advisor", requests_used=50)
        store.close()

        args = MagicMock()
        with patch("bet_advisor.main._get_sqlite_path", return_value=str(db_path)):
            with patch.dict(
                "os.environ",
                {"ODDS_API_PROJECT_SHARE": "200", "ODDS_API_PROJECT_TAG": "bet-advisor"},
            ):
                code = cmd_quota(args)

        out = capsys.readouterr().out
        assert "50" in out  # used count
        assert "200" in out  # project share
        assert code == 0


# ---------------------------------------------------------------------------
# cmd_health integration test (real SQLite, no HTTP)
# ---------------------------------------------------------------------------


class TestCmdHealth:
    def test_health_prints_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        from bet_advisor.storage.sqlite_store import SQLiteStore
        from bet_advisor.recommend.model_health import (
            ensure_model_health_table,
            record_model_health,
        )

        db_path = tmp_path / "health_test.db"
        store = SQLiteStore(db_path)
        store.connect()
        ensure_model_health_table(store)
        record_model_health(
            store,
            model_version="test-v1",
            brier=0.2500,
            ece=0.0120,
        )
        store.close()

        args = MagicMock()
        with patch("bet_advisor.main._get_sqlite_path", return_value=str(db_path)):
            code = cmd_health(args)

        out = capsys.readouterr().out
        assert "test-v1" in out
        assert "0.25" in out
        assert code == 0
