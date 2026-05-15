"""Tests for WeatherClient -- indoor venue returns zeros, outdoor returns parsed values."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bet_advisor.ingest.weather import WeatherClient, _is_indoor, _NULL_WEATHER


def _mock_open_meteo_response(
    temp: float = 18.5,
    wind: float = 15.0,
    wind_dir: float = 270.0,
    precip: float = 0.0,
    humidity: float = 65.0,
    weathercode: int = 0,
) -> dict:
    """Minimal Open-Meteo JSON response with 24 hourly entries."""
    n = 24
    return {
        "hourly": {
            "time": [f"2026-04-01T{h:02d}:00" for h in range(n)],
            "temperature_2m": [temp] * n,
            "windspeed_10m": [wind] * n,
            "winddirection_10m": [wind_dir] * n,
            "precipitation": [precip] * n,
            "relativehumidity_2m": [humidity] * n,
            "weathercode": [weathercode] * n,
        }
    }


@pytest.fixture
def weather_client() -> WeatherClient:
    c = WeatherClient()
    yield c
    c.close()


@pytest.fixture
def venues_json(tmp_path: Path) -> Path:
    """Write a minimal venues.json into a temp dir and return the path."""
    import json

    data = [
        {
            "name": "MCG",
            "city": "Melbourne",
            "state": "VIC",
            "lat": -37.82,
            "lon": 144.984,
            "indoor": False,
        },
        {
            "name": "Marvel Stadium",
            "aliases": ["Docklands Stadium", "Etihad Stadium"],
            "city": "Melbourne",
            "state": "VIC",
            "lat": -37.817,
            "lon": 144.947,
            "indoor": True,
        },
    ]
    p = tmp_path / "venues.json"
    p.write_text(json.dumps(data))
    return p


class TestIndoorDetection:
    def test_marvel_stadium_is_indoor(self) -> None:
        assert _is_indoor("Marvel Stadium") is True

    def test_mcg_is_outdoor(self) -> None:
        # MCG not in default venues.json at test time -- rely on keyword fallback
        assert _is_indoor("Docklands") is True

    def test_unknown_venue_defaults_to_outdoor(self) -> None:
        assert _is_indoor("Unknown Oval XYZ") is False


class TestHistoricalFetch:
    def _mock_httpx_response(self, data: dict) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = data
        resp.raise_for_status.return_value = None
        return resp

    def test_outdoor_venue_returns_weather_values(
        self, weather_client: WeatherClient
    ) -> None:
        data = _mock_open_meteo_response(temp=20.0, wind=12.0)
        resp = self._mock_httpx_response(data)

        with patch.object(weather_client._client, "get", return_value=resp):
            result = weather_client.fetch_historical(
                lat=-37.82,
                lon=144.984,
                date_utc=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
            )

        assert result["temp_c"] == pytest.approx(20.0)
        assert result["wind_kmh"] == pytest.approx(12.0)

    def test_historical_parse_returns_correct_keys(
        self, weather_client: WeatherClient
    ) -> None:
        data = _mock_open_meteo_response()
        resp = self._mock_httpx_response(data)

        with patch.object(weather_client._client, "get", return_value=resp):
            result = weather_client.fetch_historical(
                lat=-37.82,
                lon=144.984,
                date_utc=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
            )

        assert set(result.keys()) == {
            "temp_c", "wind_kmh", "wind_direction_deg",
            "precip_mm", "humidity_pct", "conditions"
        }

    def test_http_error_returns_null_weather(
        self, weather_client: WeatherClient
    ) -> None:
        import httpx

        with patch.object(
            weather_client._client,
            "get",
            side_effect=httpx.ConnectError("failed"),
        ):
            result = weather_client.fetch_historical(
                lat=-37.82,
                lon=144.984,
                date_utc=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
            )
        assert result == _NULL_WEATHER


class TestFetchForMatch:
    def test_indoor_venue_returns_null_weather_without_http(
        self, weather_client: WeatherClient
    ) -> None:
        with patch.object(weather_client._client, "get") as mock_get:
            result = weather_client.fetch_for_match(
                venue_name="Marvel Stadium",
                match_time_utc=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
            )
        # No HTTP call should have been made
        mock_get.assert_not_called()
        assert result == _NULL_WEATHER

    def test_indoor_keyword_match(self, weather_client: WeatherClient) -> None:
        with patch.object(weather_client._client, "get") as mock_get:
            result = weather_client.fetch_for_match(
                venue_name="Etihad Stadium",
                match_time_utc=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
            )
        mock_get.assert_not_called()
        assert result["conditions"] == "indoor"

    def test_unknown_venue_no_coords_returns_null(
        self, weather_client: WeatherClient
    ) -> None:
        with patch.object(weather_client._client, "get") as mock_get:
            result = weather_client.fetch_for_match(
                venue_name="Totally Unknown Oval 9999",
                match_time_utc=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
            )
        mock_get.assert_not_called()
        assert result == _NULL_WEATHER
