"""Tests for BetfairClient stub -- construction, interface, and mock backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bet_advisor.ingest.betfair import BetfairClient


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
