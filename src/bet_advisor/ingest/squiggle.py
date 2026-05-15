"""
Squiggle API client for AFL fixtures, results, and aggregated tipster predictions.

API docs: https://squiggle.com.au/the-squiggle-api/
No authentication required. Must set a descriptive User-Agent including contact.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
_USER_AGENT = "bet-advisor/dev (jacobemarriott@icloud.com) personal-research"
_REQUEST_DELAY = 0.5  # seconds between requests to be polite


class SquiggleClient:
    """Thin client for the Squiggle community AFL API.

    All methods return raw list-of-dict from the API response. Callers
    are responsible for mapping to their own data models.
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.Client(
            base_url=SQUIGGLE_BASE,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        self._last_request: float = 0.0

    def _get(self, params: dict[str, Any]) -> dict:
        """Perform a GET request to the Squiggle API with polite rate limiting."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)

        resp = self._client.get("", params=params)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def fetch_games(self, year: int) -> list[dict]:
        """Fetch all games (fixtures + results) for a season year.

        Returns a list of game dicts with keys including: id, year, round,
        date, tz, hteam, ateam, hteamid, ateamid, hscore, ascore, venue,
        complete, is_final.
        """
        data = self._get({"q": "games", "year": year})
        games: list[dict] = data.get("games", [])
        logger.info("Squiggle: fetched %d games for %d", len(games), year)
        return games

    def fetch_results(self, year: int, round_num: int) -> list[dict]:
        """Fetch completed results for a specific round.

        Returns the same structure as fetch_games but filtered to one round
        and only completed games (complete == 100).
        """
        data = self._get({"q": "games", "year": year, "round": round_num})
        games: list[dict] = data.get("games", [])
        results = [g for g in games if g.get("complete", 0) == 100]
        logger.info(
            "Squiggle: %d completed results for %d round %d",
            len(results),
            year,
            round_num,
        )
        return results

    def fetch_tips(self, year: int, round_num: int) -> list[dict]:
        """Fetch aggregated tipster predictions for a round.

        Includes predictions from 50+ community models. Useful as an ensemble
        'market consensus' feature in models.

        Returns list of tip dicts with keys: gameid, source (model name),
        tip (predicted winner), tipteamid, correct (if completed), margin,
        confidence, hteam, ateam, hteamid, ateamid.
        """
        data = self._get({"q": "tips", "year": year, "round": round_num})
        tips: list[dict] = data.get("tips", [])
        logger.info("Squiggle: %d tips for %d round %d", len(tips), year, round_num)
        return tips

    def fetch_sources(self) -> list[dict]:
        """List all participating prediction models."""
        data = self._get({"q": "sources"})
        return data.get("sources", [])

    def close(self) -> None:
        self._client.close()
