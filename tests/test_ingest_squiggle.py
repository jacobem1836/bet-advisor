"""Tests for SquiggleClient -- mocked HTTP, parse fixtures into matches."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from bet_advisor.ingest.squiggle import SquiggleClient


def _make_games_response(n: int = 3) -> dict:
    return {
        "games": [
            {
                "id": i,
                "year": 2026,
                "round": 1,
                "date": "2026-04-01 19:25:00",
                "tz": "AEST",
                "hteam": "Richmond",
                "ateam": "Melbourne",
                "hteamid": 14,
                "ateamid": 10,
                "hscore": 90 if i < 2 else None,
                "ascore": 80 if i < 2 else None,
                "venue": "MCG",
                "complete": 100 if i < 2 else 0,
                "is_final": 0,
            }
            for i in range(n)
        ]
    }


def _make_tips_response(n: int = 5) -> dict:
    return {
        "tips": [
            {
                "gameid": 1,
                "source": f"Model{i}",
                "tip": "Richmond",
                "tipteamid": 14,
                "correct": 1,
                "margin": 10,
                "confidence": 0.65,
                "hteam": "Richmond",
                "ateam": "Melbourne",
                "hteamid": 14,
                "ateamid": 10,
            }
            for i in range(n)
        ]
    }


@pytest.fixture
def client() -> SquiggleClient:
    c = SquiggleClient()
    yield c
    c.close()


class TestFetchGames:
    def _mock_response(self, data: dict, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = data
        if status >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=resp
            )
        else:
            resp.raise_for_status.return_value = None
        return resp

    def test_fetch_games_returns_list(self, client: SquiggleClient) -> None:
        resp = self._mock_response(_make_games_response(3))
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                games = client.fetch_games(2026)
        assert isinstance(games, list)
        assert len(games) == 3

    def test_fetch_games_fields(self, client: SquiggleClient) -> None:
        resp = self._mock_response(_make_games_response(1))
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                games = client.fetch_games(2026)
        game = games[0]
        assert game["hteam"] == "Richmond"
        assert game["ateam"] == "Melbourne"
        assert game["venue"] == "MCG"
        assert game["year"] == 2026

    def test_fetch_results_filters_completed(self, client: SquiggleClient) -> None:
        # 3 games: first is complete (100), rest are not
        resp = self._mock_response(_make_games_response(3))
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                results = client.fetch_results(2026, round_num=1)
        # Only the first game has complete==100
        assert all(g["complete"] == 100 for g in results)

    def test_fetch_games_empty_response(self, client: SquiggleClient) -> None:
        resp = self._mock_response({"games": []})
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                games = client.fetch_games(2026)
        assert games == []

    def test_fetch_tips_returns_model_tips(self, client: SquiggleClient) -> None:
        resp = self._mock_response(_make_tips_response(5))
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                tips = client.fetch_tips(2026, round_num=1)
        assert len(tips) == 5
        assert all("source" in t for t in tips)
        assert all("confidence" in t for t in tips)

    def test_http_error_propagates(self, client: SquiggleClient) -> None:
        resp = self._mock_response({}, status=500)
        with patch.object(client._client, "get", return_value=resp):
            with patch("time.sleep"):
                with pytest.raises(httpx.HTTPStatusError):
                    client.fetch_games(2026)
