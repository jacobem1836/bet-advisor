"""Tests for BetfairClient and BetfairDelayedClient with mocked backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bet_advisor.ingest.betfair import BetfairClient, BetfairDelayedClient


class TestBetfairClientConstruction:
    def test_constructs_without_certs(self) -> None:
        """BetfairClient must be instantiable without valid certs for testing."""
        client = BetfairClient(
            username="test",
            password="test",
            app_key="testkey",
            certs="",
        )
        assert client._username == "test"
        assert client._app_key == "testkey"

    def test_default_construction(self) -> None:
        client = BetfairClient()
        assert client._trading is None

    def test_afl_event_type_id(self) -> None:
        from bet_advisor.ingest.betfair import AFL_EVENT_TYPE_ID

        assert AFL_EVENT_TYPE_ID == "61420"


class TestBetfairLoginMock:
    def test_login_with_mock_backend(self) -> None:
        """login() must call betfairlightweight.APIClient.login() without raising."""
        mock_trading = MagicMock()
        mock_api_class = MagicMock(return_value=mock_trading)

        with patch.dict(
            "sys.modules",
            {"betfairlightweight": MagicMock(APIClient=mock_api_class)},
        ):
            client = BetfairClient(
                username="u",
                password="p",
                app_key="k",
                certs="/fake/path",
            )
            client.login()

        mock_trading.login.assert_called_once()

    def test_login_raises_without_credentials(self) -> None:
        client = BetfairClient()
        with pytest.raises(ValueError, match="username"):
            client.login()

    def test_call_without_login_raises(self) -> None:
        client = BetfairClient(username="u", password="p", app_key="k")
        with pytest.raises(RuntimeError, match="login"):
            client.list_afl_events()


class TestListAflEventsMock:
    def test_list_afl_events_callable_with_mock(self) -> None:
        """list_afl_events() must return a list of event dicts when backend is mocked."""
        mock_event = MagicMock()
        mock_event.event.id = "12345"
        mock_event.event.name = "Richmond v Melbourne"
        mock_event.event.country_code = "AU"
        mock_event.event.timezone = "Australia/Melbourne"
        mock_event.event.open_date = None
        mock_event.market_count = 3

        mock_trading = MagicMock()
        mock_trading.betting.list_events.return_value = [mock_event]
        mock_api_class = MagicMock(return_value=mock_trading)

        mock_filters_module = MagicMock()
        mock_filters_module.market_filter.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "betfairlightweight": MagicMock(APIClient=mock_api_class),
                "betfairlightweight.filters": mock_filters_module,
            },
        ):
            client = BetfairClient(username="u", password="p", app_key="k")
            client._trading = mock_trading  # skip real login
            events = client.list_afl_events()

        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["event_id"] == "12345"
        assert events[0]["event_name"] == "Richmond v Melbourne"
        assert events[0]["market_count"] == 3

    def test_list_afl_events_returns_empty_on_error(self) -> None:
        mock_trading = MagicMock()
        mock_trading.betting.list_events.side_effect = Exception("API error")
        mock_filters_module = MagicMock()
        mock_filters_module.market_filter.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "betfairlightweight": MagicMock(),
                "betfairlightweight.filters": mock_filters_module,
            },
        ):
            client = BetfairClient(username="u", password="p", app_key="k")
            client._trading = mock_trading
            events = client.list_afl_events()

        assert events == []


# ---------------------------------------------------------------------------
# BetfairDelayedClient tests
# ---------------------------------------------------------------------------


class TestBetfairDelayedClientConstruction:
    def test_constructs_with_explicit_params(self) -> None:
        client = BetfairDelayedClient(
            username="u",
            password="p",
            app_key="delayed_key",
            cert_path="/fake/cert.crt",
            cert_key_path="/fake/cert.key",
        )
        assert client._username == "u"
        assert client._app_key == "delayed_key"
        assert client._trading is None

    def test_falls_back_to_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BETFAIR_USERNAME", "env_user")
        monkeypatch.setenv("BETFAIR_PASSWORD", "env_pass")
        monkeypatch.setenv("BETFAIR_APP_KEY_DELAYED", "env_key")
        monkeypatch.setenv("BETFAIR_CERT_PATH", "/env/cert.crt")
        monkeypatch.setenv("BETFAIR_CERT_KEY_PATH", "/env/cert.key")
        client = BetfairDelayedClient()
        assert client._username == "env_user"
        assert client._app_key == "env_key"

    def test_is_live_is_false(self) -> None:
        client = BetfairDelayedClient(username="u", password="p", app_key="k")
        assert client.is_live is False

    def test_default_construction_trading_is_none(self) -> None:
        client = BetfairDelayedClient()
        assert client._trading is None


class TestBetfairDelayedClientLogin:
    def test_login_with_mock_backend(self) -> None:
        """login() should create an APIClient and call login() on it."""
        mock_trading = MagicMock()
        mock_api_class = MagicMock(return_value=mock_trading)

        with patch.dict(
            "sys.modules",
            {"betfairlightweight": MagicMock(APIClient=mock_api_class)},
        ):
            client = BetfairDelayedClient(
                username="u",
                password="p",
                app_key="k",
                cert_path="/c/cert.crt",
                cert_key_path="/c/cert.key",
            )
            client.login()

        mock_trading.login.assert_called_once()

    def test_login_raises_without_credentials(self) -> None:
        client = BetfairDelayedClient()
        with pytest.raises(ValueError, match="app_key"):
            client.login()

    def test_call_without_login_raises(self) -> None:
        client = BetfairDelayedClient(username="u", password="p", app_key="k")
        with pytest.raises(RuntimeError, match="login"):
            client.list_afl_events()


class TestBetfairDelayedListAflEvents:
    def test_list_afl_events_callable_with_mock(self) -> None:
        mock_event = MagicMock()
        mock_event.event.id = "99999"
        mock_event.event.name = "Collingwood v Richmond"
        mock_event.event.country_code = "AU"
        mock_event.event.timezone = "Australia/Melbourne"
        mock_event.event.open_date = None
        mock_event.market_count = 5

        mock_trading = MagicMock()
        mock_trading.betting.list_events.return_value = [mock_event]
        mock_api_class = MagicMock(return_value=mock_trading)
        mock_filters_module = MagicMock()
        mock_filters_module.market_filter.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "betfairlightweight": MagicMock(APIClient=mock_api_class),
                "betfairlightweight.filters": mock_filters_module,
            },
        ):
            client = BetfairDelayedClient(username="u", password="p", app_key="k")
            client._trading = mock_trading
            events = client.list_afl_events()

        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["event_id"] == "99999"
        assert events[0]["event_name"] == "Collingwood v Richmond"
        assert events[0]["market_count"] == 5

    def test_list_afl_events_returns_empty_on_error(self) -> None:
        mock_trading = MagicMock()
        mock_trading.betting.list_events.side_effect = Exception("API error")
        mock_filters_module = MagicMock()
        mock_filters_module.market_filter.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "betfairlightweight": MagicMock(),
                "betfairlightweight.filters": mock_filters_module,
            },
        ):
            client = BetfairDelayedClient(username="u", password="p", app_key="k")
            client._trading = mock_trading
            events = client.list_afl_events()

        assert events == []


class TestBetfairDelayedGetMarketBook:
    def _make_mock_runner(
        self,
        selection_id: int = 1,
        status: str = "ACTIVE",
        last_price_traded: float = 1.85,
        total_matched: float = 5000.0,
    ) -> MagicMock:
        r = MagicMock()
        r.selection_id = selection_id
        r.status = status
        r.last_price_traded = last_price_traded
        r.total_matched = total_matched
        return r

    def test_get_market_book_returns_parsed_dict(self) -> None:
        runner_a = self._make_mock_runner(1, "ACTIVE", 1.85, 8000.0)
        runner_b = self._make_mock_runner(2, "ACTIVE", 2.20, 7500.0)

        mock_book = MagicMock()
        mock_book.status = "OPEN"
        mock_book.total_matched = 15500.0
        mock_book.runners = [runner_a, runner_b]

        mock_trading = MagicMock()
        mock_trading.betting.list_market_book.return_value = [mock_book]
        mock_filters_module = MagicMock()
        mock_filters_module.price_projection.return_value = {}

        with patch.dict(
            "sys.modules",
            {
                "betfairlightweight": MagicMock(),
                "betfairlightweight.filters": mock_filters_module,
            },
        ):
            client = BetfairDelayedClient(username="u", password="p", app_key="k")
            client._trading = mock_trading
            result = client.get_market_book("1.12345")

        assert result["market_id"] == "1.12345"
        assert result["status"] == "OPEN"
        assert len(result["runners"]) == 2
        assert result["runners"][0]["last_price_traded"] == 1.85
        assert result["runners"][1]["last_price_traded"] == 2.20

    def test_get_market_book_raises_without_login(self) -> None:
        client = BetfairDelayedClient(username="u", password="p", app_key="k")
        with pytest.raises(RuntimeError, match="login"):
            client.get_market_book("1.12345")


class TestBetfairDelayedGetCloseSnapshot:
    """Tests for the wait_for_suspend polling logic."""

    def _build_client_with_mock(self, book_responses: list[MagicMock]) -> BetfairDelayedClient:
        """Return a client whose get_market_book() cycles through book_responses."""
        client = BetfairDelayedClient(username="u", password="p", app_key="k")
        # Mock get_market_book directly so we avoid module-level patching complexity.
        call_count = [0]

        def _mock_get_book(market_id: str) -> dict:
            idx = min(call_count[0], len(book_responses) - 1)
            call_count[0] += 1
            return book_responses[idx]

        client.get_market_book = _mock_get_book  # type: ignore[method-assign]
        return client

    def test_no_wait_returns_immediately(self) -> None:
        snapshot = {"market_id": "1.999", "status": "OPEN", "total_matched": 100.0, "runners": []}
        client = self._build_client_with_mock([snapshot])
        result = client.get_close_snapshot("1.999", wait_for_suspend=False)
        assert result["status"] == "OPEN"

    def test_wait_for_suspend_polls_until_suspended(self) -> None:
        """get_close_snapshot should poll until SUSPENDED is returned."""
        open_snap = {"market_id": "1.999", "status": "OPEN", "total_matched": 50.0, "runners": []}
        suspended_snap = {
            "market_id": "1.999",
            "status": "SUSPENDED",
            "total_matched": 95000.0,
            "runners": [
                {
                    "selection_id": 1,
                    "last_price_traded": 1.85,
                    "total_matched": 55000.0,
                    "status": "ACTIVE",
                },
                {
                    "selection_id": 2,
                    "last_price_traded": 2.10,
                    "total_matched": 40000.0,
                    "status": "ACTIVE",
                },
            ],
        }
        responses = [open_snap, open_snap, suspended_snap]
        client = self._build_client_with_mock(responses)

        with patch("time.sleep"):
            result = client.get_close_snapshot(
                "1.999",
                wait_for_suspend=True,
                timeout_s=60,
                poll_interval_s=1.0,
            )

        assert result["status"] == "SUSPENDED"
        assert result["runners"][0]["last_price_traded"] == 1.85

    def test_wait_for_suspend_returns_last_snapshot_on_timeout(self) -> None:
        """When timeout is reached without SUSPENDED, last snapshot is returned."""
        open_snap = {"market_id": "1.999", "status": "OPEN", "total_matched": 50.0, "runners": []}
        client = self._build_client_with_mock([open_snap])

        # Simulate: first monotonic() call returns 0 (deadline = timeout_s),
        # second call (loop check) returns timeout_s + 1 (expired immediately after
        # the first successful poll).
        # The loop must poll at least once before expiring.
        call_count = [0]

        def _fast_clock() -> float:
            call_count[0] += 1
            # call 1: deadline calculation  -> 0.0
            # call 2: first loop condition  -> 0.0  (still inside window)
            # call 3: second loop condition -> 9999.0 (expired)
            if call_count[0] <= 2:
                return 0.0
            return 9999.0

        with (
            patch("time.sleep"),
            patch("bet_advisor.ingest.betfair.time.monotonic", side_effect=_fast_clock),
        ):
            result = client.get_close_snapshot(
                "1.999",
                wait_for_suspend=True,
                timeout_s=1,
                poll_interval_s=0.01,
            )

        assert result["status"] == "OPEN"

    def test_is_live_remains_false(self) -> None:
        client = BetfairDelayedClient(username="u", password="p", app_key="k")
        assert client.is_live is False
