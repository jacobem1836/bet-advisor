"""
Daily markdown report renderer and writer.

Produces a dense, structured markdown card for each day's betting activity.
The report covers top recommendations, per-recommendation detail, bankroll
P&L, CLV summary, model health, risk flags, and Odds API quota.

Style rules:
- No em dashes. Use en dashes (-) and hyphens.
- Markdown tables for tabular data.
- Numbers rounded to context-appropriate precision.
- No fluff or promotional language.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bet_advisor.recommend.engine import Recommendation
    from bet_advisor.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class MarkdownReport:
    """Renders and writes daily markdown reports.

    Parameters
    ----------
    sqlite_store:
        Connected SQLiteStore for quota lookups.
    output_dir:
        Directory where reports are written. Created if absent.
    """

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        output_dir: str | Path = "reports",
    ) -> None:
        self._sqlite = sqlite_store
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public interface

    def render_daily(
        self,
        report_date: date,
        recs: list[Recommendation],
        pnl_snapshot: dict[str, Any],
        model_health: dict[str, Any],
        clv_reference_summary: dict[str, Any] | None = None,
    ) -> str:
        """Render the daily report as a markdown string.

        Parameters
        ----------
        report_date:
            The date this report covers.
        recs:
            Recommendations for this day (already filtered and capped).
        pnl_snapshot:
            Output of compute_pnl_snapshot for this date.
        model_health:
            Output of compute_model_health.

        Returns
        -------
        Complete markdown string.
        """
        sections: list[str] = []
        sections.append(self._section_header(report_date, pnl_snapshot))
        sections.append(self._section_top_recs_table(recs))
        sections.append(self._section_rec_details(recs))
        sections.append(self._section_pnl(pnl_snapshot))
        sections.append(self._section_clv_reference(clv_reference_summary))
        sections.append(self._section_model_health(model_health))
        sections.append(self._section_risk_flags(model_health, pnl_snapshot, recs))
        sections.append(self._section_quota(report_date))
        return "\n\n".join(sections)

    def write_daily(
        self,
        report_date: date,
        recs: list[Recommendation],
        pnl_snapshot: dict[str, Any],
        model_health: dict[str, Any],
        clv_reference_summary: dict[str, Any] | None = None,
    ) -> Path:
        """Render and write the daily report to disk.

        Parameters
        ----------
        report_date:
            The date this report covers.
        recs:
            Recommendations for this day.
        pnl_snapshot:
            Output of compute_pnl_snapshot.
        model_health:
            Output of compute_model_health.

        Returns
        -------
        Path to the written file.
        """
        md = self.render_daily(report_date, recs, pnl_snapshot, model_health, clv_reference_summary)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{report_date.isoformat()}.md"
        path.write_text(md, encoding="utf-8")
        logger.info("Daily report written: %s", path)
        return path

    # ------------------------------------------------------------------
    # Section renderers

    def _section_header(self, report_date: date, pnl: dict[str, Any]) -> str:
        bankroll = pnl.get("bankroll", 0.0)
        day_count = pnl.get("today_bets", 0)
        today_currency = pnl.get("today_units", 0.0)

        lines = [
            f"# Bet Advisor -- Daily Report {report_date.isoformat()}",
            "",
            f"**Bankroll:** ${bankroll:,.2f}  ",
            f"**Bets today:** {day_count}  ",
            f"**Staked today:** ${today_currency:,.2f}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} AEST",
        ]
        return "\n".join(lines)

    def _section_top_recs_table(self, recs: list[Recommendation]) -> str:
        if not recs:
            return "## Top Recommendations\n\nNo recommendations meet the minimum edge threshold today."

        header = (
            "## Top Recommendations\n\n"
            "| # | Event | Market | Runner | Bookmaker | Odds | Model P | Mkt P (devigged) | Edge | Stake | Tier |\n"
            "|---|-------|--------|--------|-----------|------|---------|-----------------|------|-------|------|\n"
        )
        rows: list[str] = []
        for i, rec in enumerate(recs, 1):
            stake_display = (
                f"{rec.recommended_stake_units:.2%} / ${rec.recommended_stake_currency:.2f}"
            )
            rows.append(
                f"| {i} "
                f"| {rec.event_id} "
                f"| {rec.market} "
                f"| {rec.runner} "
                f"| {rec.bookmaker} "
                f"| {rec.decimal_odds:.2f} "
                f"| {rec.model_prob:.3f} "
                f"| {rec.devigged_market_prob:.3f} "
                f"| {rec.edge:.3f} "
                f"| {stake_display} "
                f"| {rec.confidence_tier} |"
            )
        return header + "\n".join(rows)

    def _section_rec_details(self, recs: list[Recommendation]) -> str:
        if not recs:
            return ""

        parts = ["## Recommendation Detail"]
        for i, rec in enumerate(recs, 1):
            parts.append(f"\n### {i}. {rec.runner} ({rec.market}) @ {rec.bookmaker}")
            parts.append(
                f"- Odds: {rec.decimal_odds:.2f}  "
                f"Model P: {rec.model_prob:.3f} [{rec.model_prob_low:.3f}-{rec.model_prob_high:.3f}]  "
                f"Mkt P: {rec.devigged_market_prob:.3f}"
            )
            parts.append(
                f"- EV per unit: {rec.ev_units:.3f}  Kelly (full): {rec.kelly_fraction:.4f}  "
                f"Stake: {rec.recommended_stake_units:.2%} / ${rec.recommended_stake_currency:.2f}  "
                f"Tier: {rec.confidence_tier}"
            )

            # Rationale
            if rec.rationale:
                parts.append("\n**Rationale:**")
                for k, v in rec.rationale.items():
                    parts.append(f"- {k}: {v}")

            # Counterarguments
            if rec.counterarguments:
                parts.append("\n**Counterarguments:**")
                for arg in rec.counterarguments:
                    parts.append(f"- {arg}")

        return "\n".join(parts)

    def _section_pnl(self, pnl: dict[str, Any]) -> str:
        roi = pnl.get("roi_alltime")
        roi_str = f"{roi:.1%}" if roi is not None else "N/A"

        wl = pnl.get("roi_wilson_lower")
        wu = pnl.get("roi_wilson_upper")
        wilson_str = f"{wl:.1%} - {wu:.1%}" if wl is not None and wu is not None else "N/A"

        mean_clv = pnl.get("mean_clv")
        pct_pos = pnl.get("pct_positive_clv")
        clv_str = f"{mean_clv:.4f}" if mean_clv is not None else "N/A"
        pct_clv_str = f"{pct_pos:.1%}" if pct_pos is not None else "N/A"

        lines = [
            "## Bankroll & P&L Snapshot",
            "",
            "| Period | Bets | Staked | --- |",
            "|--------|------|--------|-----|",
            f"| Today | {pnl.get('today_bets', 0)} | ${pnl.get('today_units', 0.0):,.2f} | -- |",
            f"| 7 days | {pnl.get('week_bets', 0)} | ${pnl.get('week_units', 0.0):,.2f} | -- |",
            f"| Month | {pnl.get('month_bets', 0)} | ${pnl.get('month_units', 0.0):,.2f} | -- |",
            f"| All time | {pnl.get('alltime_bets', 0)} | ${pnl.get('alltime_units', 0.0):,.2f} | "
            f"Won ${pnl.get('alltime_won_units', 0.0):,.2f} |",
            "",
            f"**All-time ROI:** {roi_str}  **95% Wilson CI on win rate:** {wilson_str}",
            f"**Mean CLV (settled):** {clv_str}  **% Positive CLV:** {pct_clv_str}  "
            f"**n settled with CLV:** {pnl.get('n_settled', 0)}",
        ]
        return "\n".join(lines)

    def _section_clv_reference(self, clv_ref: dict[str, Any] | None) -> str:
        """Render the CLV reference summary section.

        Parameters
        ----------
        clv_ref:
            Optional dict with keys:
            - ``mode`` (str): resolver mode, e.g. ``"multi_book_consensus"``
            - ``books_used`` (list[str]): books that contributed to the reference
            - ``n_settled_with_fallback`` (int): how many settled bets used a fallback
            - ``fallback_warnings`` (list[str]): warning strings from fallback events

        If None, renders a default message showing the system default.
        """
        if clv_ref is None:
            lines = [
                "## CLV Reference",
                "",
                "CLV reference: multi_book_consensus (default) across "
                "{sportsbet, tab, ladbrokes, pointsbet, betr}. "
                "No settled bets with CLV data yet.",
            ]
            return "\n".join(lines)

        mode = clv_ref.get("mode", "multi_book_consensus")
        books = clv_ref.get("books_used", [])
        books_str = "{" + ", ".join(books) + "}" if books else "(none)"
        n_fallback = clv_ref.get("n_settled_with_fallback", 0)
        fallback_warnings = clv_ref.get("fallback_warnings", [])

        lines = [
            "## CLV Reference",
            "",
            f"CLV reference: **{mode}** across {books_str}",
        ]

        if n_fallback > 0:
            lines.append("")
            lines.append(
                f"**{n_fallback} settled bet(s) used a fallback reference source.** "
                "Fallback occurs when fewer books than the minimum are available or "
                "Betfair volume is below the configured threshold."
            )
            if fallback_warnings:
                lines.append("")
                lines.append("Fallback details:")
                for w in fallback_warnings[:5]:  # cap at 5 to keep report compact
                    lines.append(f"- {w}")
                if len(fallback_warnings) > 5:
                    lines.append(f"- ... and {len(fallback_warnings) - 5} more.")

        return "\n".join(lines)

    def _section_model_health(self, health: dict[str, Any]) -> str:
        version = health.get("last_model_version") or "unknown"
        captured = health.get("last_captured_at") or "never"
        days = health.get("days_since_snapshot")
        days_str = f"{days}d ago" if days is not None else "never"

        brier = health.get("latest_brier")
        ece = health.get("latest_ece")
        log_loss = health.get("latest_log_loss")

        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"
        ll_str = f"{log_loss:.4f}" if log_loss is not None else "N/A"

        drawdown = health.get("drawdown_pct", 0.0)
        n_bets = health.get("n_bets", 0)

        lines = [
            "## Model Health",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Model version | {version} |",
            f"| Last calibration snapshot | {captured} ({days_str}) |",
            f"| Brier score | {brier_str} |",
            f"| ECE | {ece_str} |",
            f"| Log loss | {ll_str} |",
            f"| Drawdown from peak | {drawdown:.1f}% |",
            f"| Total bets logged | {n_bets} |",
            "",
            f"**Trigger summary:** {health.get('trigger_summary', 'No active triggers')}",
        ]
        return "\n".join(lines)

    def _section_risk_flags(
        self,
        health: dict[str, Any],
        pnl: dict[str, Any],
        recs: list[Recommendation],
    ) -> str:
        flags: list[str] = []

        if health.get("ece_trigger"):
            flags.append("ECE above 0.02 -- recalibrate model before next betting cycle.")

        if health.get("drawdown_trigger"):
            flags.append(
                f"Drawdown at {health.get('drawdown_pct', 0):.1f}% -- "
                "consider reducing stakes until model is validated."
            )

        if health.get("clv_negative_trigger"):
            flags.append("Rolling 100-bet mean CLV is negative -- review model assumptions.")

        if health.get("brier_deteriorated"):
            flags.append("Brier score has deteriorated >0.02 over 60 days -- consider retraining.")

        speculative_count = sum(1 for r in recs if r.confidence_tier == "speculative")
        if speculative_count > 0:
            flags.append(
                f"{speculative_count} speculative rec(s) today -- "
                "stakes are auto-reduced 50%; verify model calibration before scaling."
            )

        if not flags:
            flags.append("No active risk flags.")

        lines = ["## Risk Flags", ""]
        for flag in flags:
            lines.append(f"- {flag}")
        return "\n".join(lines)

    def _section_quota(self, report_date: date) -> str:
        """Show Odds API MTD usage vs budget."""
        year_month = report_date.strftime("%Y-%m")
        try:
            # Try to fetch from SQLite (project_tag is not known here; use all)
            rows = self._sqlite.query(
                "SELECT SUM(requests_used) as total FROM quota_usage WHERE date LIKE ?",
                (f"{year_month}%",),
            )
            mtd_used = int(rows[0]["total"] or 0) if rows else 0
        except Exception as exc:
            logger.debug("Could not fetch quota usage: %s", exc)
            mtd_used = 0

        lines = [
            "## API Quota",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Odds API requests MTD | {mtd_used:,} |",
            f"| Period | {year_month} |",
            "",
            "Quota budget configured via `ODDS_API_PROJECT_SHARE` env var. "
            "Run `bet-advisor quota` for a live breakdown.",
        ]
        return "\n".join(lines)
