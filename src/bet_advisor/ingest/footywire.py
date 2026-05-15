"""
FootyWire scraper for advanced AFL player and team statistics.

Fetches time on ground %, metres gained, pressure acts, injury list,
and team selections. All requests are rate-limited to 1 req/sec and
raw HTML is cached to data/cache/footywire/.

Note: FootyWire has no explicit API or scraping permission. The AFL
analytics community has used it for years without documented enforcement.
Keep requests polite and do not redistribute scraped data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

FOOTYWIRE_BASE = "https://www.footywire.com"
_CACHE_DIR = Path("data/cache/footywire")
_USER_AGENT = "bet-advisor/dev (jacobemarriott@icloud.com) personal-research"
_REQUEST_DELAY = 1.0  # 1 request per second


def _ensure_cache() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


class FootyWireClient:
    """Scraper for FootyWire advanced statistics and team selections."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.Client(
            base_url=FOOTYWIRE_BASE,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        self._last_request: float = 0.0

    def _get(self, path: str, params: dict | None = None) -> str:
        """Fetch a page, respecting the 1 req/sec rate limit."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)

        resp = self._client.get(path, params=params or {})
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp.text

    def _cached_get(self, cache_key: str, path: str, params: dict | None = None) -> str:
        """Return cached HTML if available, otherwise fetch and cache."""
        cache_path = _ensure_cache() / f"{cache_key}.html"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        html = self._get(path, params)
        cache_path.write_text(html, encoding="utf-8")
        return html

    def _parse_table(self, html: str) -> list[dict]:
        """Parse the first HTML table in the response into a list of row dicts."""
        try:
            from bs4 import BeautifulSoup  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("beautifulsoup4 is required") from exc

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if not table:
            return []

        headers: list[str] = []
        rows: list[dict] = []
        for i, row in enumerate(table.find_all("tr")):  # type: ignore[union-attr]
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if i == 0:
                headers = cells
            elif cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))

        return rows

    # ------------------------------------------------------------------
    # Team advanced stats

    def fetch_team_advanced(self, year: int) -> pd.DataFrame:
        """Fetch team advanced statistics for a season.

        Covers metrics like time on ground %, metres gained, pressure acts,
        score involvements. Returns a DataFrame with raw column names from
        the FootyWire table.
        """
        cache_key = f"team_advanced_{year}"
        path = "/afl/footy/afl_statistics.cgi"
        params: dict[str, Any] = {
            "year": year,
            "view": "p",
        }
        try:
            html = self._cached_get(cache_key, path, params)
        except httpx.HTTPError as exc:
            logger.warning("FootyWire team advanced fetch failed for %d: %s", year, exc)
            return pd.DataFrame()

        rows = self._parse_table(html)
        if not rows:
            logger.warning("No team advanced data parsed for %d", year)
        df = pd.DataFrame(rows)
        df["season"] = year
        return df

    # ------------------------------------------------------------------
    # Player advanced stats

    def fetch_player_advanced(self, year: int) -> pd.DataFrame:
        """Fetch player advanced statistics for a season.

        Covers disposals, time on ground %, metres gained, pressure acts,
        SuperCoach scores, AFL Fantasy scores.
        """
        cache_key = f"player_advanced_{year}"
        path = "/afl/footy/afl_statistics.cgi"
        params: dict[str, Any] = {"year": year}
        try:
            html = self._cached_get(cache_key, path, params)
        except httpx.HTTPError as exc:
            logger.warning("FootyWire player advanced fetch failed for %d: %s", year, exc)
            return pd.DataFrame()

        rows = self._parse_table(html)
        df = pd.DataFrame(rows)
        df["season"] = year
        return df

    # ------------------------------------------------------------------
    # Injury list

    def fetch_injuries(self) -> pd.DataFrame:
        """Fetch the current AFL injury list.

        Returns a DataFrame with columns: player, team, injury, return_round.
        Raw HTML is NOT cached here because it changes frequently.
        """
        try:
            html = self._get("/afl/footy/injury_list")
        except httpx.HTTPError as exc:
            logger.warning("FootyWire injury fetch failed: %s", exc)
            return pd.DataFrame()

        rows = self._parse_table(html)
        if not rows:
            logger.warning("No injury data parsed from FootyWire")
            return pd.DataFrame()

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Team selections

    def fetch_team_selections(self, round_num: int | None = None) -> pd.DataFrame:
        """Fetch final team selections (22-man squads).

        round_num is informational -- FootyWire always shows the current round.
        Returns a DataFrame with columns: team, player, position, status.

        Raw HTML is NOT cached because selections change through the week.
        """
        try:
            html = self._get("/afl/footy/afl_team_selections")
        except httpx.HTTPError as exc:
            logger.warning("FootyWire selections fetch failed: %s", exc)
            return pd.DataFrame()

        try:
            from bs4 import BeautifulSoup  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("beautifulsoup4 is required") from exc

        soup = BeautifulSoup(html, "lxml")
        rows: list[dict] = []
        current_team = ""

        for tag in soup.find_all(["h3", "a"]):
            if tag.name == "h3":
                current_team = tag.get_text(strip=True)
            elif tag.name == "a" and current_team:
                player_name = tag.get_text(strip=True)
                if player_name:
                    rows.append(
                        {
                            "team": current_team,
                            "player": player_name,
                            "round": round_num,
                        }
                    )

        if not rows:
            logger.warning("No selection data parsed from FootyWire")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def close(self) -> None:
        self._client.close()
