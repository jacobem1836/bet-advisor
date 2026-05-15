"""
Ported and extended OddsAPIClient from match-bet.

Adds support for spreads, totals, and player prop markets. Returns typed
Pydantic models instead of raw dicts. Retries on 5xx and 429 via tenacity.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
AFL_SPORT_KEY = "aussierules_afl"

BOOKMAKERS: dict[str, str] = {
    "sportsbet": "Sportsbet",
    "tab": "TAB",
    "ladbrokes": "Ladbrokes",
    "neds": "Neds",
    "pointsbet": "PointsBet",
    "betr": "Betr",
    "bet365": "Bet365",
    "unibet": "Unibet",
    "betfair_ex_au": "Betfair Exchange AU",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runner:
    """A single selection within a market (e.g. team name, 'Over', player name)."""

    name: str
    price: float
    point: float | None = None  # Handicap or O/U line value


@dataclass(frozen=True)
class MarketLine:
    """All runners for one market from one bookmaker."""

    key: str  # e.g. 'h2h', 'spreads', 'player_disposals'
    runners: tuple[Runner, ...]
    last_update: datetime | None = None


@dataclass(frozen=True)
class BookmakerLine:
    """All markets offered by one bookmaker for one event."""

    key: str  # Canonical bookmaker key
    title: str
    markets: tuple[MarketLine, ...]
    last_update: datetime | None = None


@dataclass(frozen=True)
class EventOdds:
    """Full odds response for a single event."""

    event_id: str
    sport_key: str
    sport_title: str
    home_team: str
    away_team: str
    commence_time: datetime
    bookmakers: tuple[BookmakerLine, ...]


@dataclass
class OddsSnapshot:
    """Flattened snapshot record ready for storage."""

    event_id: str
    bookmaker: str
    market: str
    runner: str
    price: float
    point: float | None
    captured_at: datetime
    commence_time: datetime
    source: str = "odds_api"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TimeoutException)


# ---------------------------------------------------------------------------
# OddsAPIClient
# ---------------------------------------------------------------------------


class OddsAPIClient:
    """
    Wraps The Odds API v4 (https://the-odds-api.com/).

    Supports H2H, spreads, totals, and AFL player prop markets.
    Retries automatically on transient 5xx and 429 responses.
    API key is read from the ODDS_API_KEY environment variable if not passed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        resolved = api_key or os.environ.get("ODDS_API_KEY")
        if not resolved:
            raise ValueError("Odds API key must be passed or set via ODDS_API_KEY env var")
        self._api_key = resolved
        self._client = httpx.Client(timeout=timeout)
        self.remaining_quota: int | None = None

    # ------------------------------------------------------------------
    # Internal

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        params = {**params, "apiKey": self._api_key}

        @retry(
            retry=retry_if_exception(_should_retry),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _do() -> Any:
            resp = self._client.get(f"{ODDS_API_BASE}{path}", params=params)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            if remaining != "?":
                try:
                    self.remaining_quota = int(remaining)
                except ValueError:
                    pass
            logger.info("Odds API: %s | used=%s remaining=%s", path, used, remaining)
            return resp.json()

        return _do()

    @staticmethod
    def _parse_dt(iso: str) -> datetime:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # Events (raw)

    def fetch_events(
        self,
        sport_key: str = AFL_SPORT_KEY,
        regions: str = "au",
        bookmaker_keys: list[str] | None = None,
        commence_time_from: datetime | None = None,
        commence_time_to: datetime | None = None,
    ) -> list[dict]:
        """Fetch raw event dicts from The Odds API. Used by fetch_odds internally."""
        params: dict[str, Any] = {
            "regions": regions,
            "markets": "h2h",
            "bookmakers": ",".join(bookmaker_keys or list(BOOKMAKERS.keys())),
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            return self._get(f"/sports/{sport_key}/odds", params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("Sport %s not found -- skipping", sport_key)
            else:
                logger.error("Odds API error for %s: %s", sport_key, exc)
            return []
        except httpx.RequestError as exc:
            logger.error("Network error for %s: %s", sport_key, exc)
            return []

    # ------------------------------------------------------------------
    # Typed fetch

    def fetch_odds(
        self,
        sport_key: str = AFL_SPORT_KEY,
        regions: str = "au",
        markets: list[str] | None = None,
        odds_format: str = "decimal",
        bookmakers: list[str] | None = None,
        commence_time_from: datetime | None = None,
        commence_time_to: datetime | None = None,
    ) -> list[EventOdds]:
        """Fetch odds for a sport and return typed EventOdds models.

        Parameters
        ----------
        markets:
            List of market keys, e.g. ['h2h', 'totals', 'spreads'].
            Defaults to ['h2h', 'totals', 'spreads'].
        """
        market_keys = markets or ["h2h", "totals", "spreads"]
        bk_keys = bookmakers or list(BOOKMAKERS.keys())

        params: dict[str, Any] = {
            "regions": regions,
            "markets": ",".join(market_keys),
            "bookmakers": ",".join(bk_keys),
            "dateFormat": "iso",
            "oddsFormat": odds_format,
        }
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            raw = self._get(f"/sports/{sport_key}/odds", params)
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("fetch_odds failed: %s", exc)
            return []

        return [self._parse_event(e) for e in raw]

    def fetch_player_props(
        self,
        sport_key: str = AFL_SPORT_KEY,
        event_id: str = "",
        markets: list[str] | None = None,
    ) -> list[EventOdds]:
        """Fetch player prop odds for a single event.

        The Odds API requires event_id for player prop endpoints.
        """
        prop_markets = markets or [
            "player_disposals",
            "player_goals_anytime",
            "player_marks",
            "player_tackles",
        ]
        params: dict[str, Any] = {
            "regions": "au",
            "markets": ",".join(prop_markets),
            "dateFormat": "iso",
            "oddsFormat": "decimal",
        }
        try:
            raw = self._get(f"/sports/{sport_key}/events/{event_id}/odds", params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("Event %s not found for props", event_id)
            else:
                logger.error("fetch_player_props failed: %s", exc)
            return []
        except httpx.RequestError as exc:
            logger.error("Network error in fetch_player_props: %s", exc)
            return []

        # Single event response -- wrap in list for consistent return type
        if isinstance(raw, dict):
            return [self._parse_event(raw)]
        return [self._parse_event(e) for e in raw]

    # ------------------------------------------------------------------
    # Parsing

    def _parse_event(self, event: dict) -> EventOdds:
        bookmakers: list[BookmakerLine] = []
        for bk_data in event.get("bookmakers", []):
            markets_out: list[MarketLine] = []
            for market in bk_data.get("markets", []):
                runners = tuple(
                    Runner(
                        name=o["name"],
                        price=float(o["price"]),
                        point=float(o["point"])
                        if "point" in o and o["point"] is not None
                        else None,
                    )
                    for o in market.get("outcomes", [])
                )
                last_update = None
                if lu := market.get("last_update"):
                    last_update = self._parse_dt(lu)
                markets_out.append(
                    MarketLine(
                        key=market["key"],
                        runners=runners,
                        last_update=last_update,
                    )
                )
            bk_lu = None
            if lu := bk_data.get("last_update"):
                bk_lu = self._parse_dt(lu)
            bookmakers.append(
                BookmakerLine(
                    key=bk_data["key"],
                    title=bk_data.get("title", bk_data["key"]),
                    markets=tuple(markets_out),
                    last_update=bk_lu,
                )
            )
        return EventOdds(
            event_id=event["id"],
            sport_key=event["sport_key"],
            sport_title=event.get("sport_title", ""),
            home_team=event.get("home_team", ""),
            away_team=event.get("away_team", ""),
            commence_time=self._parse_dt(event["commence_time"]),
            bookmakers=tuple(bookmakers),
        )

    # ------------------------------------------------------------------
    # Flattened snapshots

    def flatten_snapshots(
        self,
        events: list[EventOdds],
        captured_at: datetime | None = None,
    ) -> list[OddsSnapshot]:
        """Convert EventOdds list into flat OddsSnapshot records for storage."""
        ts = captured_at or datetime.now(timezone.utc)
        rows: list[OddsSnapshot] = []
        for event in events:
            for bk in event.bookmakers:
                for market in bk.markets:
                    for runner in market.runners:
                        rows.append(
                            OddsSnapshot(
                                event_id=event.event_id,
                                bookmaker=bk.key,
                                market=market.key,
                                runner=runner.name,
                                price=runner.price,
                                point=runner.point,
                                captured_at=ts,
                                commence_time=event.commence_time,
                            )
                        )
        return rows

    def close(self) -> None:
        self._client.close()
