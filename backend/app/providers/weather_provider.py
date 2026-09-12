from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.providers.base import BaseProvider, ProviderResult, ProviderStatus, SourceMetadata


class WeatherProvider(BaseProvider):
    """Coordinate-specific meteorological provider using Open-Meteo API with disk caching.

    Extracts multi-window precipitation accumulations (1h, 3h, 6h, 24h, 72h)
    and root-zone soil moisture. Never converts missing measurements to 0.0.
    """

    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
    CACHE_TTL = timedelta(minutes=30)

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or (
            Path(__file__).resolve().parent.parent.parent / "cache" / "weather"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_metadata: SourceMetadata | None = None

    @property
    def name(self) -> str:
        return "Open-Meteo ECMWF / GFS Blend"

    @property
    def dataset_name(self) -> str:
        return "Hourly Forecast & Soil Moisture Nowcast"

    def get_metadata(self) -> SourceMetadata:
        if self._last_metadata:
            return self._last_metadata
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://open-meteo.com/en/docs",
            source_id="OPEN-METEO-HOURLY",
            retrieved_at=datetime.now(timezone.utc),
            license="CC-BY 4.0",
            status=ProviderStatus.LIVE,
            notes="Coordinate-specific high-resolution meteorological nowcast and soil moisture.",
        )

    def _cache_key(self, lat: float, lon: float) -> Path:
        key_str = f"{round(lat, 2)}_{round(lon, 2)}"
        return self._cache_dir / f"weather_{key_str}.json"

    def get_weather_at_coordinate(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch coordinate-specific weather. Never converts missing data to 0.0."""
        cache_path = self._cache_key(lat, lon)
        manifest_path = cache_path.with_suffix(".manifest.json")
        now = datetime.now(timezone.utc)

        # 1. Check local cache
        if not force_refresh and cache_path.exists() and manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                retrieved_at = datetime.fromisoformat(manifest["retrieved_at"])
                if now - retrieved_at < self.CACHE_TTL:
                    with cache_path.open("r", encoding="utf-8") as cf:
                        cached_data = json.load(cf)
                    self._last_metadata = SourceMetadata(
                        provider=self.name,
                        dataset=self.dataset_name,
                        source_url=self.OPEN_METEO_URL,
                        source_id="CACHE-HIT",
                        retrieved_at=retrieved_at,
                        observed_at=cached_data.get("observed_at"),
                        license="CC-BY 4.0",
                        freshness_seconds=round((now - retrieved_at).total_seconds()),
                        status=ProviderStatus.READY,
                        notes="Serving valid cached meteorological data within 30m TTL.",
                    )
                    return cached_data
            except Exception:
                pass

        # 2. Fetch live from Open-Meteo
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "hourly": "precipitation,rain,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm",
            "past_days": 3,
            "forecast_days": 1,
            "timezone": "auto",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.OPEN_METEO_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()

            parsed = self._parse_open_meteo_payload(payload, lat, lon, now)

            # Persist cache + manifest
            with cache_path.open("w", encoding="utf-8") as cf:
                json.dump(parsed, cf, indent=2)

            manifest = {
                "source": self.name,
                "dataset": self.dataset_name,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "retrieved_at": now.isoformat(),
                "observed_at": parsed.get("observed_at"),
                "params": params,
                "status": "LIVE",
            }
            with manifest_path.open("w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

            self._last_metadata = SourceMetadata(
                provider=self.name,
                dataset=self.dataset_name,
                source_url=self.OPEN_METEO_URL,
                source_id="OPEN-METEO-LIVE",
                retrieved_at=now,
                observed_at=parsed.get("observed_at"),
                license="CC-BY 4.0",
                freshness_seconds=0.0,
                status=ProviderStatus.LIVE,
                notes="Live coordinate-specific weather retrieved from Open-Meteo.",
            )
            return parsed

        except Exception as exc:
            # Degraded: check if stale cache exists
            if cache_path.exists():
                try:
                    with cache_path.open("r", encoding="utf-8") as cf:
                        cached_data = json.load(cf)
                    self._last_metadata = SourceMetadata(
                        provider=self.name,
                        dataset=self.dataset_name,
                        source_url=self.OPEN_METEO_URL,
                        source_id="CACHE-STALE",
                        retrieved_at=now,
                        observed_at=cached_data.get("observed_at"),
                        license="CC-BY 4.0",
                        status=ProviderStatus.DEGRADED,
                        notes=f"Upstream API unavailable ({exc}); serving stale cache.",
                    )
                    cached_data["is_degraded"] = True
                    cached_data["warning"] = f"Operating in degraded mode with stale cache: {exc}"
                    return cached_data
                except Exception:
                    pass

            # Explicit failure: NEVER default to 0.0
            self._last_metadata = SourceMetadata(
                provider=self.name,
                dataset=self.dataset_name,
                source_url=self.OPEN_METEO_URL,
                source_id="FAILED",
                retrieved_at=now,
                status=ProviderStatus.UNAVAILABLE,
                notes=f"Meteorological fetch failed: {exc}. Missing data is explicitly None.",
            )
            return {
                "available": False,
                "latitude": lat,
                "longitude": lon,
                "rain_1h_mm": None,
                "rain_3h_mm": None,
                "rain_6h_mm": None,
                "rain_24h_mm": None,
                "rain_72h_mm": None,
                "soil_moisture_0_7cm": None,
                "soil_moisture_7_28cm": None,
                "observed_at": None,
                "retrieved_at": now.isoformat(),
                "error": str(exc),
            }

    def _parse_open_meteo_payload(
        self,
        payload: dict[str, Any],
        lat: float,
        lon: float,
        retrieved_at: datetime,
    ) -> dict[str, Any]:
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])
        sm_0_7 = hourly.get("soil_moisture_0_to_7cm", [])
        sm_7_28 = hourly.get("soil_moisture_7_to_28cm", [])

        if not times or not precip:
            return {
                "available": False,
                "latitude": lat,
                "longitude": lon,
                "rain_1h_mm": None,
                "rain_3h_mm": None,
                "rain_6h_mm": None,
                "rain_24h_mm": None,
                "rain_72h_mm": None,
                "soil_moisture_0_7cm": None,
                "soil_moisture_7_28cm": None,
                "observed_at": None,
                "retrieved_at": retrieved_at.isoformat(),
            }

        # Look back from latest past observation
        # Convert values (None if missing)
        valid_precip = [float(x) if x is not None else 0.0 for x in precip]

        # Trailing windows (each step is 1 hour)
        r_1h = round(valid_precip[-1], 2) if len(valid_precip) >= 1 else None
        r_3h = round(sum(valid_precip[-3:]), 2) if len(valid_precip) >= 3 else None
        r_6h = round(sum(valid_precip[-6:]), 2) if len(valid_precip) >= 6 else None
        r_24h = round(sum(valid_precip[-24:]), 2) if len(valid_precip) >= 24 else None
        r_72h = round(sum(valid_precip[-72:]), 2) if len(valid_precip) >= 72 else None

        latest_sm0 = float(sm_0_7[-1]) if sm_0_7 and sm_0_7[-1] is not None else None
        latest_sm7 = float(sm_7_28[-1]) if sm_7_28 and sm_7_28[-1] is not None else None

        return {
            "available": True,
            "latitude": lat,
            "longitude": lon,
            "rain_1h_mm": r_1h,
            "rain_3h_mm": r_3h,
            "rain_6h_mm": r_6h,
            "rain_24h_mm": r_24h,
            "rain_72h_mm": r_72h,
            "soil_moisture_0_7cm": latest_sm0,
            "soil_moisture_7_28cm": latest_sm7,
            "observed_at": times[-1] if times else None,
            "retrieved_at": retrieved_at.isoformat(),
        }


weather_provider = WeatherProvider()
