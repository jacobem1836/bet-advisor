"""
Betfair Exchange client for AFL market data and closing line capture.

Phase 2: login() and list_afl_events() are implemented using betfairlightweight.
Streaming API hookup is deferred to Phase 9.

Requires:
- A verified Betfair AU account
- An App Key from https://developer.betfair.com/
- SSL certificates (self-signed, generated locally)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AFL_EVENT_TYPE_ID = "61420"


class BetfairClient:
    """Betfair Exchange API client for AFL markets.

    Authentication uses betfairlightweight with cert-based login.

    Usage:
        client = BetfairClient(
            username="...",
            password="...",
            app_key="...",
            certs="/path/to/certs",
        )
        client.login()
        events = client.list_afl_events()
        client.logout()
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        app_key: str = "",
        certs: str | Path = "",
        locale: str = "australia",
    ) -> None:
        self._username = username
        self._password = password
        self._app_key = app_key
        self._certs = str(certs)
        self._locale = locale
        self._trading: Any = None

    # ------------------------------------------------------------------
    # Auth

    def login(self) -> None:
        """Authenticate with Betfair Exchange via cert-based login.

        Raises RuntimeError if betfairlightweight is not installed or
        if credentials/certs are missing.
        """
        try:
            import betfairlightweight  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "betfairlightweight is required: pip install betfairlightweight"
            ) from exc

        if not all([self._username, self._password, self._app_key]):
            raise ValueError("username, password, and app_key must all be set before login")

        self._trading = betfairlightweight.APIClient(
            username=self._username,
            password=self._password,
            app_key=self._app_key,
            certs=self._certs or None,
            locale=self._locale,
        )
        self._trading.login()
        logger.info("Betfair login successful for %s", self._username)

    def logout(self) -> None:
        """Log out from Betfair Exchange."""
        if self._trading is not None:
            try:
                self._trading.logout()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Betfair logout error: %s", exc)
            self._trading = None

    # ------------------------------------------------------------------
    # Market data

    def list_afl_events(self, max_results: int = 50) -> list[dict]:
        """List upcoming AFL events from the Betfair Exchange.

        Returns a list of event summary dicts with keys:
        event_id, event_name, country_code, timezone, open_date, market_count.
        """
        if self._trading is None:
            raise RuntimeError("Call login() before list_afl_events()")

        try:
            import betfairlightweight.filters as bf_filters  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("betfairlightweight is required") from exc

        event_filter = bf_filters.market_filter(
            event_type_ids=[AFL_EVENT_TYPE_ID],
        )
        try:
            events = self._trading.betting.list_events(
                filter=event_filter,
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("list_afl_events failed: %s", exc)
            return []

        result = []
        for container in events:
            ev = container.event
            result.append(
                {
                    "event_id": ev.id,
                    "event_name": ev.name,
                    "country_code": ev.country_code,
                    "timezone": ev.timezone,
                    "open_date": ev.open_date.isoformat() if ev.open_date else None,
                    "market_count": container.market_count,
                }
            )
        logger.info("list_afl_events: %d events returned", len(result))
        return result

    def list_match_odds_markets(self, max_results: int = 50) -> list[dict]:
        """List upcoming AFL MATCH_ODDS markets.

        Returns lightweight market catalogue entries with market_id,
        market_name, total_matched, and runners list.
        """
        if self._trading is None:
            raise RuntimeError("Call login() before list_match_odds_markets()")

        try:
            import betfairlightweight.filters as bf_filters  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("betfairlightweight is required") from exc

        market_filter = bf_filters.market_filter(
            event_type_ids=[AFL_EVENT_TYPE_ID],
            market_types=["MATCH_ODDS"],
        )
        try:
            markets = self._trading.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=["RUNNER_DESCRIPTION", "MARKET_START_TIME"],
                max_results=max_results,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("list_match_odds_markets failed: %s", exc)
            return []

        result = []
        for m in markets:
            result.append(
                {
                    "market_id": m.market_id,
                    "market_name": m.market_name,
                    "total_matched": m.total_matched,
                    "start_time": (
                        m.market_start_time.isoformat() if m.market_start_time else None
                    ),
                    "runners": [
                        {
                            "selection_id": r.selection_id,
                            "runner_name": r.runner_name,
                            "handicap": r.handicap,
                        }
                        for r in (m.runners or [])
                    ],
                }
            )
        return result

    # Phase 9 -- Betfair Streaming API hookup
    # The streaming API is the preferred approach for closing line capture.
    # Uncomment and implement in Phase 9:
    #
    # def start_stream(self, market_ids: list[str]) -> None:
    #     """Subscribe to Betfair Streaming API for real-time price updates."""
    #     # Phase 9
    #     raise NotImplementedError("Betfair streaming is implemented in Phase 9")
