"""
AFL Tables ingestion -- historical match results and player statistics.

Primary source: afltables.com via the pyAFL library.
Falls back to a thin custom scraper if pyAFL is unavailable or returns empty.

Cached HTML/JSON is written to data/cache/afltables/ so the full history
does not need to be re-fetched on every run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache/afltables")
_REQUEST_DELAY = 1.5  # seconds between requests -- be polite


def _ensure_cache() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _try_pyafl_matches(start_year: int, end_year: int) -> pd.DataFrame | None:
    """Attempt to load match history via pyAFL. Returns None if unavailable."""
    try:
        # pyAFL public API varies by version; attempt the most common patterns
        from pyAFL import AFL  # type: ignore[import]

        frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            try:
                season = AFL(year)
                if hasattr(season, "get_games"):
                    df = season.get_games()
                elif hasattr(season, "games"):
                    df = pd.DataFrame(season.games)
                else:
                    logger.debug("pyAFL season object has no games method for %d", year)
                    continue
                if df is not None and not df.empty:
                    df["season"] = year
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pyAFL failed for year %d: %s", year, exc)
        return pd.concat(frames, ignore_index=True) if frames else None
    except ImportError:
        logger.info("pyAFL not importable -- falling back to scraper")
        return None


def _scrape_matches_year(year: int, session: Any) -> pd.DataFrame:
    """Scrape AFL Tables match data for one year using requests + BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for scraping") from exc

    cache_path = _ensure_cache() / f"matches_{year}.html"
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        url = f"https://afltables.com/afl/seas/{year}.html"
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        html = resp.text
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(_REQUEST_DELAY)

    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    # AFL Tables season pages use a table-based layout; parse the game rows
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) >= 6:
                rows.append(
                    {
                        "raw_row": cells,
                        "season": year,
                    }
                )

    if not rows:
        return pd.DataFrame()

    # Return raw rows for caller to normalise -- full parsing is complex
    return pd.DataFrame(rows)


def fetch_match_history(
    start_year: int = 2009,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Fetch AFL match history for a range of seasons.

    Tries pyAFL first, then falls back to direct scraping with HTML caching.
    Columns in the output DataFrame will include at minimum:
    match_id, season, round, date, venue, home_team, away_team,
    home_score, away_score, completed.

    Parameters
    ----------
    start_year:
        First season to fetch. Defaults to 2009 (when odds data begins).
    end_year:
        Last season (inclusive). Defaults to the current season year.
    """
    import datetime

    if end_year is None:
        end_year = datetime.date.today().year

    # Attempt pyAFL first
    df = _try_pyafl_matches(start_year, end_year)
    if df is not None and not df.empty:
        logger.info(
            "fetch_match_history: pyAFL returned %d rows for %d-%d",
            len(df),
            start_year,
            end_year,
        )
        return _normalise_matches(df)

    # Fallback: scrape directly
    logger.info("Using AFL Tables scraper for %d-%d", start_year, end_year)
    import requests  # type: ignore[import]

    session = requests.Session()
    session.headers["User-Agent"] = "bet-advisor/dev (jacobemarriott@icloud.com) personal-research"

    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        try:
            df_year = _scrape_matches_year(year, session)
            if not df_year.empty:
                frames.append(df_year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scrape failed for %d: %s", year, exc)

    if not frames:
        logger.warning("fetch_match_history returned no data")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("fetch_match_history: scraped %d raw rows", len(combined))
    return _normalise_matches(combined)


def _normalise_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the matches schema used by DuckDBStore."""
    # Column name mapping from common pyAFL column names
    rename_map = {
        "Game": "match_id",
        "game": "match_id",
        "Round": "round",
        "Date": "date",
        "Venue": "venue",
        "Home.team": "home_team",
        "Hteam": "home_team",
        "Away.team": "away_team",
        "Ateam": "away_team",
        "Home.score": "home_score",
        "Hscore": "home_score",
        "Away.score": "away_score",
        "Ascore": "away_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in [
        "match_id",
        "season",
        "round",
        "date",
        "venue",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]:
        if col not in df.columns:
            df[col] = None

    if "completed" not in df.columns:
        # Mark as completed if both scores are present
        df["completed"] = df["home_score"].notna() & df["away_score"].notna()

    return df[
        [
            "match_id",
            "season",
            "round",
            "date",
            "venue",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "completed",
        ]
    ]


def fetch_player_history(
    start_year: int = 2012,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Fetch AFL player match statistics for a range of seasons.

    Returns a DataFrame with columns matching the player_stats schema:
    match_id, player_id, player_name, team, position, time_on_ground_pct,
    disposals, kicks, handballs, marks, tackles, goals, behinds,
    clearances, fantasy_points, supercoach_points.

    Time on ground is only available from FootyWire (2012+). This function
    returns AFL Tables data which lacks that column -- it will be None.
    """
    import datetime

    if end_year is None:
        end_year = datetime.date.today().year

    try:
        from pyAFL import AFL  # type: ignore[import]

        frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            try:
                season = AFL(year)
                if hasattr(season, "get_player_stats"):
                    df = season.get_player_stats()
                elif hasattr(season, "player_stats"):
                    df = pd.DataFrame(season.player_stats)
                else:
                    continue
                if df is not None and not df.empty:
                    df["season"] = year
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.debug("pyAFL player stats failed for %d: %s", year, exc)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            logger.info("fetch_player_history: pyAFL returned %d rows", len(combined))
            return _normalise_player_stats(combined)
    except ImportError:
        logger.info("pyAFL not importable for player history")

    logger.warning(
        "fetch_player_history: pyAFL unavailable and scraper not implemented -- "
        "returning empty DataFrame"
    )
    return pd.DataFrame()


def _normalise_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to the player_stats schema."""
    rename_map = {
        "Game": "match_id",
        "ID": "player_id",
        "Player": "player_name",
        "Team": "team",
        "Position": "position",
        "Disposals": "disposals",
        "Kicks": "kicks",
        "Handballs": "handballs",
        "Marks": "marks",
        "Tackles": "tackles",
        "Goals": "goals",
        "Behinds": "behinds",
        "Clearances": "clearances",
        "Fantasy": "fantasy_points",
        "SC": "supercoach_points",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in [
        "match_id",
        "player_id",
        "player_name",
        "team",
        "position",
        "time_on_ground_pct",
        "disposals",
        "kicks",
        "handballs",
        "marks",
        "tackles",
        "goals",
        "behinds",
        "clearances",
        "fantasy_points",
        "supercoach_points",
    ]:
        if col not in df.columns:
            df[col] = None

    return df[
        [
            "match_id",
            "player_id",
            "player_name",
            "team",
            "position",
            "time_on_ground_pct",
            "disposals",
            "kicks",
            "handballs",
            "marks",
            "tackles",
            "goals",
            "behinds",
            "clearances",
            "fantasy_points",
            "supercoach_points",
        ]
    ]
