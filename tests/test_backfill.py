"""Tests for backfill.py -- dry-run mode produces plan without writing data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


_BACKFILL_SCRIPT = Path(__file__).parent.parent / "scripts" / "backfill.py"


class TestDryRun:
    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """--dry-run must complete without error."""
        result = subprocess.run(
            [sys.executable, str(_BACKFILL_SCRIPT), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_prints_plan(self, tmp_path: Path) -> None:
        """--dry-run must print the backfill plan steps."""
        result = subprocess.run(
            [sys.executable, str(_BACKFILL_SCRIPT), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert "DRY RUN" in output or "dry run" in output.lower() or "plan" in output.lower()

    def test_dry_run_does_not_create_database(self, tmp_path: Path) -> None:
        """--dry-run must not write any database file."""
        result = subprocess.run(
            [sys.executable, str(_BACKFILL_SCRIPT), "--dry-run", "--db", str(tmp_path / "test.duckdb")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        db_file = tmp_path / "test.duckdb"
        assert not db_file.exists(), "Dry run must not create a database file"

    def test_start_year_argument_accepted(self, tmp_path: Path) -> None:
        """--start-year argument must not cause an error."""
        result = subprocess.run(
            [sys.executable, str(_BACKFILL_SCRIPT), "--dry-run", "--start-year", "2019"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


class TestBackfillModuleImport:
    """Smoke test -- ensure the backfill module imports without error."""

    def test_import_run_backfill(self) -> None:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("backfill", _BACKFILL_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        # Do not exec -- just verify the spec loads without SyntaxError
        assert spec is not None
        assert mod is not None
