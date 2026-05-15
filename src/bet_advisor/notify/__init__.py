"""
Notification package.

Re-exports MarkdownReport and provides a convenience function for writing
a daily markdown card from structured inputs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bet_advisor.notify.cli_report import MarkdownReport

if TYPE_CHECKING:
    from bet_advisor.recommend.engine import Recommendation
    from bet_advisor.storage.sqlite_store import SQLiteStore

__all__ = ["MarkdownReport", "write_markdown_card"]


def write_markdown_card(
    report_date: date,
    recs: list[Recommendation],
    pnl_snapshot: dict[str, Any],
    model_health: dict[str, Any],
    sqlite_store: SQLiteStore,
    output_dir: str | Path = "reports",
) -> Path:
    """Convenience wrapper: render and write a daily markdown card.

    Parameters
    ----------
    report_date:
        The date this report covers.
    recs:
        Recommendations for the day.
    pnl_snapshot:
        Output of compute_pnl_snapshot.
    model_health:
        Output of compute_model_health.
    sqlite_store:
        Connected SQLiteStore (used for quota lookup inside the report).
    output_dir:
        Output directory for the report file.

    Returns
    -------
    Path to the written file.
    """
    reporter = MarkdownReport(sqlite_store=sqlite_store, output_dir=output_dir)
    return reporter.write_daily(report_date, recs, pnl_snapshot, model_health)
