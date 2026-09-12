from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707
CACHE_TTL = timedelta(minutes=10)

_cache: dict[str, Any] | None = None
_lock = Lock()


def get_live_rainfall() -> dict[str, Any]:
    global _cache
    now = datetime.now(timezone.utc)
    with _lock:
        if _cache and now - _cache["fetched_at"] < CACHE_TTL:
            return dict(_cache)

    params = {
        "latitude": CHENNAI_LAT,
        "longitude": CHENNAI_LON,
        "hourly": "precipitation,rain",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "auto",
    }

    try:
        with httpx.Client(timeout=12.0) as client:
            response = client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            payload = response.json()
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        precipitation = [float(x or 0.0) for x in hourly.get("precipitation", [])]
        now_local = datetime.now().astimezone().replace(tzinfo=None)
        cutoff = now_local - timedelta(hours=24)
        recent = []
        for stamp, amount in zip(times, precipitation):
            try:
                parsed = datetime.fromisoformat(str(stamp))
            except ValueError:
                continue
            if cutoff <= parsed <= now_local:
                recent.append(amount)
        rainfall_24h = round(sum(recent), 2)
        observed_at = times[-1] if times else None
        result = {
            "available": True,
            "source": "Open-Meteo",
            "latitude": CHENNAI_LAT,
            "longitude": CHENNAI_LON,
            "rainfall_24h_mm": rainfall_24h,
            "observed_at": observed_at,
            "fetched_at": now,
        }
    except Exception as exc:
        result = {
            "available": False,
            "source": "Open-Meteo",
            "latitude": CHENNAI_LAT,
            "longitude": CHENNAI_LON,
            "rainfall_24h_mm": None,
            "observed_at": None,
            "fetched_at": now,
            "error": str(exc),
        }

    with _lock:
        _cache = result
    return dict(result)
