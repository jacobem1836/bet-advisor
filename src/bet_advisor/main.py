"""
CLI entrypoint for bet-advisor.

Commands
--------
bet-advisor schedule [--dry-run]
    Start the APScheduler pipeline. Use --dry-run to print the planned schedule
    without starting the scheduler.

bet-advisor recommend --round N [--persist] [--allow-untrained]
    Generate recommendations for AFL round N. Prints to stdout.
    Use --persist to write signals and bets to SQLite.
    Use --allow-untrained to bypass the untrained model guard (smoke testing only).

bet-advisor report --date YYYY-MM-DD [--output-dir DIR]
    Render and print the markdown report for the given date.

bet-advisor backtest [--config PATH]
    Re-run the walk-forward backtest (shells out to scripts/run_backtest.py).

bet-advisor quota
    Print Odds API MTD usage vs budget.

bet-advisor health
    Print model health summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sqlite_path() -> str:
    return os.environ.get("SQLITE_PATH", "data/operational.db")


def _get_duckdb_path() -> str:
    return os.environ.get("DUCKDB_PATH", "data/analytics.duckdb")


def _get_bankroll() -> float:
    return float(os.environ.get("BANKROLL", "1000.0"))


def _get_project_tag() -> str:
    return os.environ.get("ODDS_API_PROJECT_TAG", "bet-advisor")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_schedule(args: argparse.Namespace) -> int:
    """Start the scheduler (or dry-run it)."""
    from bet_advisor.scheduler import Scheduler

    scheduler = Scheduler(dry_run=args.dry_run)
    scheduler.start()
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Generate recommendations for a given AFL round."""
    from bet_advisor.recommend.engine import RecommendationConfig, RecommendationEngine
    from bet_advisor.storage.duckdb_store import DuckDBStore
    from bet_advisor.storage.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(_get_sqlite_path())
    sqlite.connect()

    try:
        duckdb = DuckDBStore(_get_duckdb_path())
        duckdb.connect()
    except Exception as exc:
        logger.warning("DuckDB unavailable: %s -- continuing without feature store", exc)
        duckdb = None  # type: ignore[assignment]

    try:
        # Use a stub model -- in production, load a fitted DisposalsModel
        model = _load_model_or_stub()
        config = RecommendationConfig(
            bankroll=_get_bankroll(),
            stake_mode=os.environ.get("STAKE_MODE", "flat"),
        )
        engine = RecommendationEngine(
            model=model, sqlite_store=sqlite, duckdb_store=duckdb, config=config
        )

        recs = engine.generate_for_round(
            round_number=args.round,
            allow_untrained=getattr(args, "allow_untrained", False),
        )

        if not recs:
            print("No recommendations meet the edge threshold for this round.")
            return 0

        is_trained = getattr(model, "is_trained", False)
        if not is_trained and getattr(args, "allow_untrained", False):
            print(
                "WARNING: model is not trained. These recommendations are for "
                "smoke-testing only and should NOT be persisted without --allow-untrained "
                "acknowledgement.",
                file=sys.stderr,
            )

        # Print recommendations
        print(f"\nRecommendations for Round {args.round} -- {len(recs)} bets\n")
        print(
            f"{'#':<4} {'Event':<20} {'Market':<20} {'Runner':<20} {'Odds':>6} {'Edge':>7} {'Stake%':>8} {'Tier':<12}"
        )
        print("-" * 95)
        for i, rec in enumerate(recs, 1):
            print(
                f"{i:<4} {rec.event_id:<20} {rec.market:<20} {rec.runner:<20} "
                f"{rec.decimal_odds:>6.2f} {rec.edge:>7.3f} {rec.recommended_stake_units:>8.2%} {rec.confidence_tier:<12}"
            )

        if args.persist:
            if not is_trained and not getattr(args, "allow_untrained", False):
                print(
                    "\nERROR: Cannot persist recommendations from an untrained model.\n"
                    "Pass --allow-untrained to override this guard.",
                    file=sys.stderr,
                )
                return 1
            bet_ids = engine.persist(recs)
            print(f"\nPersisted {len(bet_ids)} bets: IDs {bet_ids}")

    finally:
        sqlite.close()
        if duckdb is not None:
            duckdb.close()

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Render and print the markdown report for a given date."""
    try:
        report_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: invalid date format {args.date!r}. Use YYYY-MM-DD.", file=sys.stderr)
        return 1

    from bet_advisor.notify import write_markdown_card
    from bet_advisor.recommend.model_health import compute_model_health
    from bet_advisor.recommend.pnl import compute_pnl_snapshot
    from bet_advisor.storage.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(_get_sqlite_path())
    sqlite.connect()
    try:
        pnl = compute_pnl_snapshot(sqlite, as_of=report_date, bankroll=_get_bankroll())
        health = compute_model_health(sqlite, duckdb_store=None)
        output_dir = getattr(args, "output_dir", "reports") or "reports"
        path = write_markdown_card(
            report_date=report_date,
            recs=[],
            pnl_snapshot=pnl,
            model_health=health,
            sqlite_store=sqlite,
            output_dir=output_dir,
        )
        content = path.read_text(encoding="utf-8")
        print(content)
        print(f"\nReport written to: {path}")
    finally:
        sqlite.close()

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the walk-forward backtest."""
    script = Path(__file__).parent.parent.parent / "scripts" / "run_backtest.py"
    if not script.exists():
        print(f"ERROR: backtest script not found at {script}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(script)]
    config_path = getattr(args, "config", None)
    if config_path:
        cmd.extend(["--config", config_path])

    result = subprocess.run(cmd, check=False)
    return result.returncode


def cmd_quota(args: argparse.Namespace) -> int:
    """Print Odds API MTD usage vs budget."""
    from bet_advisor.storage.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(_get_sqlite_path())
    sqlite.connect()
    try:
        project_tag = _get_project_tag()
        year_month = datetime.now(UTC).strftime("%Y-%m")
        used = sqlite.get_month_to_date_usage(project_tag, year_month)
        budget = int(os.environ.get("ODDS_API_PROJECT_SHARE", "200"))
        monthly = int(os.environ.get("ODDS_API_MONTHLY_BUDGET", "500"))
        remaining = max(0, budget - used)
        print(f"\nOdds API Quota -- {year_month}")
        print(f"  Project tag:     {project_tag}")
        print(f"  Monthly budget:  {monthly:,} requests")
        print(f"  Project share:   {budget:,} requests")
        print(f"  MTD used:        {used:,}")
        print(f"  Remaining:       {remaining:,}")
        if used >= budget:
            print("  Status:          EXHAUSTED")
        else:
            pct = used / budget * 100 if budget > 0 else 0
            print(f"  Usage:           {pct:.1f}%")
    finally:
        sqlite.close()

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Print model health summary."""
    from bet_advisor.recommend.model_health import compute_model_health, ensure_model_health_table
    from bet_advisor.storage.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(_get_sqlite_path())
    sqlite.connect()
    try:
        ensure_model_health_table(sqlite)
        health = compute_model_health(sqlite, duckdb_store=None)
        print("\nModel Health Summary")
        print(f"  Version:           {health.get('last_model_version') or 'unknown'}")
        print(f"  Last snapshot:     {health.get('last_captured_at') or 'never'}")
        days = health.get("days_since_snapshot")
        print(f"  Days since snap:   {days if days is not None else 'N/A'}")
        print(f"  Brier:             {health.get('latest_brier') or 'N/A'}")
        print(f"  ECE:               {health.get('latest_ece') or 'N/A'}")
        print(f"  Log loss:          {health.get('latest_log_loss') or 'N/A'}")
        print(f"  Drawdown:          {health.get('drawdown_pct', 0.0):.1f}%")
        print(f"  Total bets:        {health.get('n_bets', 0)}")
        print()
        print(f"  Trigger summary:   {health.get('trigger_summary', 'None')}")
        print()
        active_triggers = []
        if health.get("ece_trigger"):
            active_triggers.append("ECE_HIGH")
        if health.get("drawdown_trigger"):
            active_triggers.append("DRAWDOWN_HIGH")
        if health.get("clv_negative_trigger"):
            active_triggers.append("CLV_NEGATIVE")
        if health.get("brier_deteriorated"):
            active_triggers.append("BRIER_DETERIORATED")
        if active_triggers:
            print(f"  Active triggers:   {', '.join(active_triggers)}")
        else:
            print("  Active triggers:   None")
    finally:
        sqlite.close()

    return 0


# ---------------------------------------------------------------------------
# Model loader stub
# ---------------------------------------------------------------------------


def _load_model_or_stub() -> object:
    """Load a fitted DisposalsModel from the default path, or return a stub.

    Returns an object that satisfies the BettingModel protocol.
    The stub marks itself as untrained so the engine's guard fires correctly.
    """
    model_path = os.environ.get("MODEL_PATH", "models/disposals_latest.joblib")
    if Path(model_path).exists():
        try:
            from bet_advisor.models.disposals import DisposalsModel

            model = DisposalsModel.load(model_path)
            logger.info("Loaded DisposalsModel from %s", model_path)
            return model
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", model_path, exc)

    logger.warning(
        "No trained model found at %s. Using stub model (untrained guard will fire).",
        model_path,
    )
    return _UntrainedStub()


class _UntrainedStub:
    """Stub model that is always untrained -- triggers the guard in the engine."""

    @property
    def is_trained(self) -> bool:
        return False

    @property
    def version_hash(self) -> str:
        return "stub-untrained"

    def predict_over_under_prob(self, X: object, line: float, calibrate: bool = True) -> list:
        return []


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="bet-advisor",
        description="AFL betting advisor -- statistical model pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # schedule
    p_schedule = subparsers.add_parser("schedule", help="Start the scheduler.")
    p_schedule.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned schedule without starting.",
    )

    # recommend
    p_recommend = subparsers.add_parser("recommend", help="Generate recommendations for a round.")
    p_recommend.add_argument("--round", type=int, required=True, help="AFL round number.")
    p_recommend.add_argument(
        "--persist",
        action="store_true",
        help="Write signals and bets to SQLite.",
    )
    p_recommend.add_argument(
        "--allow-untrained",
        action="store_true",
        dest="allow_untrained",
        help="Bypass the untrained model guard (smoke testing only).",
    )

    # report
    p_report = subparsers.add_parser("report", help="Render the markdown report for a date.")
    p_report.add_argument("--date", required=True, help="Date in YYYY-MM-DD format.")
    p_report.add_argument(
        "--output-dir", default="reports", help="Output directory for the report."
    )

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Run the walk-forward backtest.")
    p_backtest.add_argument("--config", default=None, help="Path to backtest config file.")

    # quota
    subparsers.add_parser("quota", help="Print Odds API MTD usage vs budget.")

    # health
    subparsers.add_parser("health", help="Print model health summary.")

    return parser


def cli() -> None:
    """Main CLI entrypoint registered in pyproject.toml."""
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "schedule": cmd_schedule,
        "recommend": cmd_recommend,
        "report": cmd_report,
        "backtest": cmd_backtest,
        "quota": cmd_quota,
        "health": cmd_health,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    cli()
