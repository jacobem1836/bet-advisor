"""
Open-Meteo weather client for AFL venue weather data.

Uses the ERA5 historical archive for past matches and the forecast API for
upcoming fixtures. No API key required for non-commercial use.

Indoor venues (Marvel Stadium) return zero/null weather features because
weather is irrelevant when the roof is closed.

Archive API: https://archive-api.open-meteo.com/v1/archive
Forecast API: https://api.open-meteo.com/v1/forecast
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
_VENUES_CONFIG = Path("config/venues.json")

_WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "winddirection_10m",
    "relativehumidity_2m",
    "weathercode",
]

_NULL_WEATHER: dict[str, Any] = {
    "temp_c": None,
    "wind_kmh": None,
    "wind_direction_deg": None,
    "precip_mm": None,
    "humidity_pct": None,
    "conditions": "indoor",
}


def _load_venues() -> dict[str, Any]:
    """Load venue metadata from config/venues.json."""
    if not _VENUES_CONFIG.exists():
        logger.warning("venues.json not found at %s", _VENUES_CONFIG)
        return {}
    with _VENUES_CONFIG.open() as f:
        venues_list: list[dict] = json.load(f)
    # Index by lowercased name for lookup
    return {v["name"].lower(): v for v in venues_list}


def _is_indoor(venue_name: str) -> bool:
    """Return True if this venue is indoor (weather irrelevant)."""
    venues = _load_venues()
    venue_key = venue_name.lower()
    for key, data in venues.items():
        if venue_key in key or key in venue_key:
            return bool(data.get("indoor", False))
    # Known indoor venues by keyword fallback
    indoor_keywords = ["marvel", "docklands", "etihad"]
    return any(kw in venue_key for kw in indoor_keywords)


def _venue_coords(venue_name: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a venue, or None if not found."""
    venues = _load_venues()
    venue_key = venue_name.lower()
    for key, data in venues.items():
        if venue_key in key or key in venue_key:
            lat = data.get("lat") or data.get("latitude")
            lon = data.get("lon") or data.get("longitude")
            if lat and lon:
                return float(lat), float(lon)
    return None


def _parse_hourly(data: dict, target_hour_utc: int) -> dict[str, Any]:
    """Extract weather values for the closest available hour."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return _NULL_WEATHER

    # Find index matching target hour (format: '2023-04-01T14:00')
    best_idx = 0
    best_diff = float("inf")
    for i, t in enumerate(times):
        try:
            hour = int(t[11:13])
            diff = abs(hour - target_hour_utc)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        except (ValueError, IndexError):
            continue

    def _get(key: str) -> Any:
        vals = hourly.get(key, [])
        return vals[best_idx] if best_idx < len(vals) else None

    return {
        "temp_c": _get("temperature_2m"),
        "wind_kmh": _get("windspeed_10m"),
        "wind_direction_deg": _get("winddirection_10m"),
        "precip_mm": _get("precipitation"),
        "humidity_pct": _get("relativehumidity_2m"),
        "conditions": str(_get("weathercode") or ""),
    }


class WeatherClient:
    """Open-Meteo client for AFL venue weather.

    Indoor venues (Marvel Stadium) return null weather features automatically.
    """

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.Client(timeout=timeout)

    def fetch_historical(
        self,
        lat: float,
        lon: float,
        date_utc: datetime,
    ) -> dict[str, Any]:
        """Fetch ERA5 historical weather for a specific date and location.

        Returns a dict with keys: temp_c, wind_kmh, wind_direction_deg,
        precip_mm, humidity_pct, conditions.
        """
        date_str = date_utc.strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": ",".join(_WEATHER_VARS),
            "timezone": "UTC",
        }
        try:
            resp = self._client.get(_ARCHIVE_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Historical weather fetch failed: %s", exc)
            return _NULL_WEATHER

        return _parse_hourly(data, target_hour_utc=date_utc.hour)

    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        when_utc: datetime,
    ) -> dict[str, Any]:
        """Fetch weather forecast for an upcoming match time and location.

        Uses Open-Meteo's hourly forecast API (up to 16 days ahead).
        """
        date_str = when_utc.strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(_WEATHER_VARS),
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC",
        }
        try:
            resp = self._client.get(_FORECAST_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Forecast weather fetch failed: %s", exc)
            return _NULL_WEATHER

        return _parse_hourly(data, target_hour_utc=when_utc.hour)

    def fetch_for_match(
        self,
        venue_name: str,
        match_time_utc: datetime,
    ) -> dict[str, Any]:
        """Fetch weather for a match, handling indoor venues automatically.

        If the venue is indoor, returns null weather features without making
        any HTTP request.

        If the match is historical (more than 5 days ago), uses the ERA5
        archive. Otherwise uses the forecast endpoint.
        """
        if _is_indoor(venue_name):
            logger.debug("Indoor venue %s -- returning null weather", venue_name)
            return _NULL_WEATHER

        coords = _venue_coords(venue_name)
        if coords is None:
            logger.warning(
                "No coordinates found for venue %r -- returning null weather",
                venue_name,
            )
            return _NULL_WEATHER

        lat, lon = coords
        now_utc = datetime.now(timezone.utc)
        age_days = (now_utc - match_time_utc.replace(tzinfo=timezone.utc)).days

        if age_days > 5:
            return self.fetch_historical(lat, lon, match_time_utc)
        return self.fetch_forecast(lat, lon, match_time_utc)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Module-level convenience functions (standalone, no client instance needed)
# ---------------------------------------------------------------------------


def fetch_historical(lat: float, lon: float, date_utc: datetime) -> dict[str, Any]:
    """Convenience wrapper for single historical fetch."""
    client = WeatherClient()
    try:
        return client.fetch_historical(lat, lon, date_utc)
    finally:
        client.close()


def fetch_forecast(lat: float, lon: float, when_utc: datetime) -> dict[str, Any]:
    """Convenience wrapper for single forecast fetch."""
    client = WeatherClient()
    try:
        return client.fetch_forecast(lat, lon, when_utc)
    finally:
        client.close()
