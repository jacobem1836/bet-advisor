#!/usr/bin/env python3
"""
Historical data backfill orchestrator.

Pulls data from all free sources and loads into DuckDB:
- AFL Tables match and player history (2009 to current)
- FootyWire advanced stats (2012 to current)
- Open-Meteo historical weather for every past match venue + date

AusSportsBetting XLSX and Betfair AU CSVs must be downloaded manually
(see README for URLs) and placed in data/imports/ before running.

Usage:
    python scripts/backfill.py [--dry-run] [--start-year 2019]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Add src to path so we can import bet_advisor without pip install -e
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from bet_advisor.ingest.afl_tables import fetch_match_history, fetch_player_history
from bet_advisor.ingest.footywire import FootyWireClient
from bet_advisor.ingest.weather import WeatherClient
from bet_advisor.storage.duckdb_store import DuckDBStore

console = Console()
logger = logging.getLogger("backfill")

_DEFAULT_START_YEAR = 2009
_DEFAULT_DB = Path("data/analytics.duckdb")
_VENUES_CONFIG = Path("config/venues.json")


def load_venues() -> list[dict]:
    if not _VENUES_CONFIG.exists():
        logger.warning("venues.json not found -- skipping weather backfill")
        return []
    with _VENUES_CONFIG.open() as f:
        return json.load(f)


def venue_coords(venue_name: str, venues: list[dict]) -> tuple[float, float] | None:
    key = venue_name.lower()
    for v in venues:
        v_key = v["name"].lower()
        aliases = [a.lower() for a in v.get("aliases", [])]
        if key in v_key or v_key in key or key in aliases:
            lat = v.get("lat") or v.get("latitude")
            lon = v.get("lon") or v.get("longitude")
            if lat and lon:
                return float(lat), float(lon)
    return None


def is_indoor(venue_name: str, venues: list[dict]) -> bool:
    key = venue_name.lower()
    for v in venues:
        v_key = v["name"].lower()
        aliases = [a.lower() for a in v.get("aliases", [])]
        if key in v_key or v_key in key or key in aliases:
            return bool(v.get("indoor", False))
    return "marvel" in key or "docklands" in key or "etihad" in key


def run_backfill(
    start_year: int,
    end_year: int,
    dry_run: bool,
    db_path: Path,
) -> None:
    """Execute the full backfill sequence."""
    console.rule(f"[bold]AFL Data Backfill {start_year}-{end_year}[/bold]")

    if dry_run:
        console.print("[yellow]DRY RUN -- no data will be written[/yellow]")

    plan = [
        f"AFL Tables matches {start_year}-{end_year}",
        f"AFL Tables player stats {start_year}-{end_year}",
        f"FootyWire advanced player stats 2012-{end_year}",
        "Open-Meteo historical weather for all outdoor match venues",
        "AusSportsBetting XLSX (manual import -- check data/imports/)",
        "Betfair AU CSVs (manual import -- check data/imports/)",
    ]

    console.print("\n[bold]Backfill plan:[/bold]")
    for i, step in enumerate(plan, 1):
        console.print(f"  {i}. {step}")

    if dry_run:
        console.print("\n[green]Dry run complete -- plan printed above[/green]")
        return

    venues = load_venues()

    with DuckDBStore(db_path) as store:
        store.init_schema()

        # Step 1 -- Match history
        console.print("\n[bold]Step 1:[/bold] Fetching AFL Tables match history...")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("AFL Tables matches...", total=None)
            matches_df = fetch_match_history(start_year, end_year)
            progress.update(task, completed=True)

        if matches_df.empty:
            console.print("[yellow]No match data returned -- check AFL Tables access[/yellow]")
        else:
            n = store.upsert_matches(matches_df)
            console.print(f"  Inserted {n} new match rows ({len(matches_df)} fetched)")

        # Step 2 -- Player history
        console.print("\n[bold]Step 2:[/bold] Fetching AFL Tables player history...")
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("AFL Tables player stats...", total=None)
            player_df = fetch_player_history(start_year, end_year)
            progress.update(task, completed=True)

        if player_df.empty:
            console.print("[yellow]No player data returned[/yellow]")
        else:
            n = store.upsert_player_stats(player_df)
            console.print(f"  Inserted {n} new player stat rows")

        # Step 3 -- FootyWire advanced
        console.print("\n[bold]Step 3:[/bold] Fetching FootyWire advanced player stats...")
        fw_client = FootyWireClient()
        try:
            fw_start = max(start_year, 2012)
            for year in range(fw_start, end_year + 1):
                console.print(f"  FootyWire {year}...")
                fw_df = fw_client.fetch_player_advanced(year)
                if not fw_df.empty:
                    console.print(f"    {len(fw_df)} rows fetched (stored in cache)")
        finally:
            fw_client.close()

        # Step 4 -- Weather backfill for completed outdoor matches
        console.print("\n[bold]Step 4:[/bold] Backfilling weather for completed matches...")
        matches_stored = store.query(
            "SELECT match_id, date, venue FROM matches WHERE completed = TRUE AND date IS NOT NULL"
        )

        if matches_stored.empty:
            console.print("[yellow]No completed matches in DB -- skipping weather[/yellow]")
        else:
            wx_client = WeatherClient()
            inserted_wx = 0
            skipped_indoor = 0
            skipped_no_coords = 0
            errors = 0

            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()
            ) as progress:
                task = progress.add_task(
                    f"Weather for {len(matches_stored)} matches...", total=None
                )
                for _, row in matches_stored.iterrows():
                    venue = str(row.get("venue", "") or "")
                    date_val = row.get("date")

                    if not venue or not date_val:
                        continue

                    if is_indoor(venue, venues):
                        skipped_indoor += 1
                        continue

                    coords = venue_coords(venue, venues)
                    if coords is None:
                        skipped_no_coords += 1
                        continue

                    try:
                        import pandas as pd

                        date_dt = pd.Timestamp(date_val).to_pydatetime()
                        lat, lon = coords
                        wx = wx_client.fetch_historical(lat, lon, date_dt)
                        wx_df = pd.DataFrame(
                            [
                                {
                                    "match_id": row["match_id"],
                                    **wx,
                                }
                            ]
                        )
                        inserted_wx += store.upsert_weather(wx_df)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Weather fetch error for %s: %s", row["match_id"], exc)
                        errors += 1

            wx_client.close()
            progress.update(task, completed=True)
            console.print(
                f"  Weather: {inserted_wx} inserted, "
                f"{skipped_indoor} indoor skipped, "
                f"{skipped_no_coords} no-coords skipped, "
                f"{errors} errors"
            )

    # Manual import reminders
    console.print("\n[bold yellow]Manual steps required:[/bold yellow]")
    console.print(
        "  AusSportsBetting: download from "
        "https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/ "
        "and place in data/imports/aussportsbetting_afl.xlsx"
    )
    console.print(
        "  Betfair AU CSVs: download from "
        "https://betfair-datascientists.github.io/data/dataListing/ "
        "and place in data/imports/betfair_au_afl/"
    )
    console.print("\n[green]Backfill complete[/green]")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="AFL historical data backfill")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the backfill plan without fetching or writing any data",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=_DEFAULT_START_YEAR,
        help=f"First season year to fetch (default: {_DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"Path to DuckDB analytics file (default: {_DEFAULT_DB})",
    )
    args = parser.parse_args()

    end_year = datetime.date.today().year

    run_backfill(
        start_year=args.start_year,
        end_year=end_year,
        dry_run=args.dry_run,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
