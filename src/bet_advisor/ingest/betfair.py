"""
Betfair Exchange client for AFL market data and closing line capture.

Two client classes are provided:

BetfairClient (Phase 2, unchanged)
-----------------------------------
General-purpose Betfair Exchange client supporting authenticated requests
via betfairlightweight. Retained for compatibility.

BetfairDelayedClient (Phase 5.5)
----------------------------------
Uses the Betfair free delayed app key to read closing prices from AFL
exchange markets without placing bets.

Key properties of the delayed key:
- Cost: free (requires a verified Betfair AU account).
- Price delay: 1-180 seconds per snapshot (variable; not a fixed interval).
- Cannot place bets: placeOrders returns PERMISSION_DENIED.
- Can read: lastPriceTraded, market status (OPEN/SUSPENDED/CLOSED).
- Market access: all AFL markets including event type 61420. Some specialised
  prop markets (e.g. first goalscorer) have historically returned access
  errors under the delayed key; the client logs and raises a clear error
  when a market is inaccessible.
- Volume data: totalMatched is absent or unreliable under the delayed key;
  BetfairDelayedClient reads it from listMarketBook but treats it as
  approximate.

Closing-price technique:
  Poll listMarketBook until market status becomes SUSPENDED (or wait_for_suspend=False
  for the most recent snapshot). The lastPriceTraded value at that point is the
  exchange closing price. The 1-180 s delay means the captured price may be up
  to 3 minutes stale in the worst case, which is still more accurate than a
  single AU recreational book's devigged close on liquid match-odds markets.

Registration:
  Sign up at https://developer.betfair.com
  Select "Delayed" app key type (not "Live"; the live key costs ~AUD 1000).
  Set BETFAIR_APP_KEY_DELAYED in your .env file.

Deprecation note:
  The legacy env var BETFAIR_APP_KEY is deprecated. Use
  BETFAIR_APP_KEY_DELAYED for read-only closing-price work and reserve
  BETFAIR_APP_KEY (live) for the execution path in Phase 9.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AFL_EVENT_TYPE_ID = "61420"

# Default poll interval in seconds when waiting for SUSPENDED status.
_DEFAULT_POLL_INTERVAL_S: float = 5.0


# ---------------------------------------------------------------------------
# Phase 2: BetfairClient (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 5.5: BetfairDelayedClient
# ---------------------------------------------------------------------------


class BetfairDelayedClient:
    """Betfair Exchange client using the free delayed app key.

    Reads closing prices from AFL exchange markets by polling
    listMarketBook until the market reaches SUSPENDED status (or on demand).
    Cannot place bets -- placeOrders returns PERMISSION_DENIED with this key.

    Price delay: 1-180 seconds per snapshot (variable; Betfair's documented
    range). In practice the stream jitters unpredictably within this band.
    A closing snapshot captured via wait_for_suspend=True may be up to 3
    minutes stale at worst. This is acceptable for match-odds closing line
    benchmarking but should be treated with caution for thin prop markets
    where liquidity is low and price discovery is weak.

    Markets where the delayed key has been known to fail:
    - Some specialised prop exchange markets (e.g. first goalscorer) may
      return access errors. The client logs the error and raises a clear
      RuntimeError when this occurs.
    - Match-odds markets (event type 61420, MATCH_ODDS type) are reliably
      accessible.

    Environment variables (loaded if parameters not passed directly):
    - BETFAIR_USERNAME
    - BETFAIR_PASSWORD
    - BETFAIR_APP_KEY_DELAYED    (the free delayed key; not the live ~AUD 1000 key)
    - BETFAIR_CERT_PATH          (path to .crt file)
    - BETFAIR_CERT_KEY_PATH      (path to .key file; passed as certs tuple)

    Parameters
    ----------
    username:
        Betfair account username. Falls back to BETFAIR_USERNAME env var.
    password:
        Betfair account password. Falls back to BETFAIR_PASSWORD env var.
    app_key:
        Delayed app key. Falls back to BETFAIR_APP_KEY_DELAYED env var.
    cert_path:
        Path to the .crt certificate file.
        Falls back to BETFAIR_CERT_PATH env var.
    cert_key_path:
        Path to the .key file. Falls back to BETFAIR_CERT_KEY_PATH env var.
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        app_key: str = "",
        cert_path: str = "",
        cert_key_path: str = "",
    ) -> None:
        self._username = username or os.environ.get("BETFAIR_USERNAME", "")
        self._password = password or os.environ.get("BETFAIR_PASSWORD", "")
        self._app_key = app_key or os.environ.get("BETFAIR_APP_KEY_DELAYED", "")
        self._cert_path = cert_path or os.environ.get("BETFAIR_CERT_PATH", "")
        self._cert_key_path = cert_key_path or os.environ.get("BETFAIR_CERT_KEY_PATH", "")
        self._trading: Any = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def is_live(self) -> bool:
        """Always False -- this client uses the free delayed key.

        The delayed key cannot place bets. For live order placement,
        the Phase 9 client uses the live key (~AUD 1000 one-off fee).
        """
        return False

    # ------------------------------------------------------------------
    # Auth

    def login(self) -> None:
        """Authenticate with Betfair Exchange via cert-based login.

        Uses the delayed app key. Raises RuntimeError if
        betfairlightweight is not installed or credentials are missing.
        """
        try:
            import betfairlightweight  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "betfairlightweight is required: pip install betfairlightweight"
            ) from exc

        if not all([self._username, self._password, self._app_key]):
            raise ValueError(
                "username, password, and app_key (BETFAIR_APP_KEY_DELAYED) "
                "must all be set before login"
            )

        # Build certs argument: betfairlightweight accepts a directory path or
        # a (cert_path, key_path) tuple.
        if self._cert_path and self._cert_key_path:
            certs_arg: Any = (self._cert_path, self._cert_key_path)
        elif self._cert_path:
            # Assume cert_path is a directory containing both files.
            certs_arg = self._cert_path
        else:
            certs_arg = None

        self._trading = betfairlightweight.APIClient(
            username=self._username,
            password=self._password,
            app_key=self._app_key,
            certs=certs_arg,
            locale="australia",
        )
        self._trading.login()
        logger.info(
            "BetfairDelayedClient login successful for %s (delayed key; cannot place bets)",
            self._username,
        )

    def logout(self) -> None:
        """Log out from Betfair Exchange."""
        if self._trading is not None:
            try:
                self._trading.logout()
            except Exception as exc:  # noqa: BLE001
                logger.warning("BetfairDelayedClient logout error: %s", exc)
            self._trading = None

    # ------------------------------------------------------------------
    # Market discovery

    def list_afl_events(self, max_results: int = 50) -> list[dict]:
        """List upcoming AFL events.

        Returns a list of dicts with keys: event_id, event_name,
        country_code, timezone, open_date, market_count.

        Raises RuntimeError if login() has not been called.
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
            logger.error("BetfairDelayedClient.list_afl_events failed: %s", exc)
            return []

        result: list[dict] = []
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
        logger.info("BetfairDelayedClient.list_afl_events: %d events", len(result))
        return result

    # ------------------------------------------------------------------
    # Market book

    def get_market_book(self, market_id: str) -> dict:
        """Fetch the current market book for a single market.

        Returns a dict with keys:
        - ``market_id`` (str)
        - ``status`` (str): one of OPEN, SUSPENDED, CLOSED
        - ``total_matched`` (float): approximate (delayed key has degraded volume)
        - ``runners`` (list[dict]): each with selection_id, name (if available),
          last_price_traded, status, total_matched

        Raises RuntimeError if login() has not been called or if the market
        is inaccessible (e.g. some prop markets under the delayed key).
        """
        if self._trading is None:
            raise RuntimeError("Call login() before get_market_book()")

        try:
            import betfairlightweight.filters as bf_filters  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("betfairlightweight is required") from exc

        price_projection = bf_filters.price_projection(
            price_data=["LAST_PRICE_TRADED"],
        )
        try:
            books = self._trading.betting.list_market_book(
                market_ids=[market_id],
                price_projection=price_projection,
            )
        except Exception as exc:
            error_msg = str(exc)
            if "MARKET_NOT_OPEN_FOR_BSP_BETTING" in error_msg or "ACCESS_DENIED" in error_msg:
                raise RuntimeError(
                    f"Market {market_id!r} is inaccessible under the delayed app key. "
                    "This can occur with specialised prop markets (e.g. first goalscorer). "
                    f"Original error: {exc}"
                ) from exc
            logger.error("get_market_book(%r) failed: %s", market_id, exc)
            raise

        if not books:
            raise RuntimeError(
                f"listMarketBook returned no data for market {market_id!r}. "
                "The market may not exist or may be inaccessible under the delayed key."
            )

        book = books[0]
        runners_out: list[dict] = []
        for runner in book.runners or []:
            runners_out.append(
                {
                    "selection_id": runner.selection_id,
                    "status": runner.status,
                    "last_price_traded": runner.last_price_traded,
                    "total_matched": runner.total_matched,
                }
            )

        return {
            "market_id": market_id,
            "status": book.status,
            "total_matched": book.total_matched,
            "runners": runners_out,
        }

    def get_close_snapshot(
        self,
        market_id: str,
        *,
        wait_for_suspend: bool = False,
        timeout_s: int = 120,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> dict:
        """Capture the closing price snapshot for a market.

        If ``wait_for_suspend=False`` (default): returns the most recent
        market book immediately.

        If ``wait_for_suspend=True``: polls ``listMarketBook`` at
        ``poll_interval_s`` intervals until the market status becomes
        SUSPENDED (kick-off / in-play) or CLOSED (settled). Returns the
        final snapshot at that point. If ``timeout_s`` is reached without
        the market suspending, returns the most recent snapshot with a
        warning logged.

        The delayed key has a 1-180 s price lag. The "closing" price captured
        just before/at SUSPENDED is the exchange close, but may be up to 3
        minutes stale. This is tolerable for match-odds benchmarking.

        Parameters
        ----------
        market_id:
            Betfair market ID (e.g. "1.234567890").
        wait_for_suspend:
            If True, poll until SUSPENDED status.
        timeout_s:
            Maximum seconds to wait when wait_for_suspend=True.
        poll_interval_s:
            Seconds between polls.

        Returns
        -------
        dict with keys:
        - ``market_id`` (str)
        - ``status`` (str)
        - ``total_matched`` (float)
        - ``runners`` (list[dict]): each with selection_id, last_price_traded,
          total_matched, status

        Raises RuntimeError if the market cannot be read.
        """
        if not wait_for_suspend:
            return self.get_market_book(market_id)

        # Polling loop.
        deadline = time.monotonic() + timeout_s
        last_snapshot: dict | None = None

        while time.monotonic() < deadline:
            snapshot = self.get_market_book(market_id)
            last_snapshot = snapshot

            status = snapshot.get("status", "")
            if status in ("SUSPENDED", "CLOSED"):
                logger.info(
                    "Market %r reached %s status; capturing closing snapshot.",
                    market_id,
                    status,
                )
                return snapshot

            logger.debug(
                "Market %r status=%s; next poll in %.1fs.",
                market_id,
                status,
                poll_interval_s,
            )
            time.sleep(poll_interval_s)

        logger.warning(
            "Market %r did not reach SUSPENDED within %ds. "
            "Returning most recent snapshot (status=%s). "
            "Closing price may be pre-close.",
            market_id,
            timeout_s,
            last_snapshot.get("status") if last_snapshot else "unknown",
        )
        if last_snapshot is None:
            raise RuntimeError(
                f"No snapshot could be obtained for market {market_id!r} within {timeout_s}s."
            )
        return last_snapshot
