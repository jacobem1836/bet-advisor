"""
Smoke test for scripts/run_backtest.py.

Verifies that the --synthetic flag produces a BacktestReport and the script
exits with code 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = str(Path(__file__).parent.parent / "scripts" / "run_backtest.py")


class TestRunBacktestScript:
    def test_synthetic_exits_zero(self, tmp_path: pytest.TempPathFactory) -> None:
        """--synthetic flag should run end-to-end and exit 0."""
        result = subprocess.run(
            [
                sys.executable,
                _SCRIPT,
                "--synthetic",
                "--output-dir",
                str(tmp_path),
                "--start-season",
                "2019",
                "--end-season",
                "2022",
                "--min-train-seasons",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_synthetic_produces_report_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Report JSON file must be written to the output directory."""
        result = subprocess.run(
            [
                sys.executable,
                _SCRIPT,
                "--synthetic",
                "--output-dir",
                str(tmp_path),
                "--start-season",
                "2019",
                "--end-season",
                "2022",
                "--min-train-seasons",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) >= 1
        report_files = list(tmp_path.glob("*/report.json"))
        assert len(report_files) >= 1

    def test_synthetic_prints_report_summary(self, tmp_path: pytest.TempPathFactory) -> None:
        """Script stdout must include the report header."""
        result = subprocess.run(
            [
                sys.executable,
                _SCRIPT,
                "--synthetic",
                "--output-dir",
                str(tmp_path),
                "--start-season",
                "2019",
                "--end-season",
                "2022",
                "--min-train-seasons",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert "Disposals Backtest Report" in result.stdout

    def test_line_strategy_bucketed(self, tmp_path: pytest.TempPathFactory) -> None:
        """--line-strategy bucketed should also exit 0."""
        result = subprocess.run(
            [
                sys.executable,
                _SCRIPT,
                "--synthetic",
                "--line-strategy",
                "bucketed",
                "--output-dir",
                str(tmp_path),
                "--start-season",
                "2019",
                "--end-season",
                "2022",
                "--min-train-seasons",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0
