from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata


class HydrologyProvider(BaseProvider):
    """Provides modelled river discharge and hydrological runoff indicators.

    Explicitly labeled as modelled / reanalysis hydrology (Copernicus GloFAS / Open-Meteo Flood API).
    Never claims to be a physical physical stream gauge unless verified.
    """

    FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"
    CACHE_TTL = timedelta(hours=6)

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or (
            Path(__file__).resolve().parent.parent.parent / "cache" / "hydrology"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_metadata: SourceMetadata | None = None

    @property
    def name(self) -> str:
        return "Copernicus GloFAS / Open-Meteo Flood API"

    @property
    def dataset_name(self) -> str:
        return "Modelled River Discharge Reanalysis & Forecast"

    def get_metadata(self) -> SourceMetadata:
        if self._last_metadata:
            return self._last_metadata
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://flood-api.open-meteo.com/",
            source_id="GLOFAS-MODELLED",
            retrieved_at=datetime.now(timezone.utc),
            license="Copernicus Open Access / CC-BY 4.0",
            status=ProviderStatus.LIVE,
            notes="Global hydrological simulation model (GloFAS). Values represent modelled catchment discharge, not a physical stream gauge.",
        )

    def _cache_key(self, lat: float, lon: float) -> Path:
        key_str = f"{round(lat, 2)}_{round(lon, 2)}"
        return self._cache_dir / f"hydro_{key_str}.json"

    def get_hydrology_at_coordinate(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
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
                        source_url=self.FLOOD_API_URL,
                        source_id="CACHE-HIT",
                        retrieved_at=retrieved_at,
                        observed_at=cached_data.get("observed_at"),
                        license="CC-BY 4.0",
                        freshness_seconds=round((now - retrieved_at).total_seconds()),
                        status=ProviderStatus.READY,
                        notes="Serving valid cached hydrological simulation within 6h TTL.",
                    )
                    return cached_data
            except Exception:
                pass

        # 2. Fetch from Flood API
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "daily": "river_discharge,river_discharge_mean",
            "forecast_days": 1,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.FLOOD_API_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()

            daily = payload.get("daily", {})
            times = daily.get("time", [])
            discharge = daily.get("river_discharge", [])
            mean_discharge = daily.get("river_discharge_mean", [])

            latest_discharge = float(discharge[0]) if discharge and discharge[0] is not None else None
            baseline_mean = float(mean_discharge[0]) if mean_discharge and mean_discharge[0] is not None else None
            anomaly = None
            if latest_discharge is not None and baseline_mean is not None and baseline_mean > 0:
                anomaly = round((latest_discharge - baseline_mean) / baseline_mean, 3)

            result = {
                "available": True,
                "data_type": "modelled_reanalysis_discharge",
                "is_physical_gauge": False,
                "latitude": lat,
                "longitude": lon,
                "river_discharge_m3s": latest_discharge,
                "baseline_mean_m3s": baseline_mean,
                "discharge_anomaly_ratio": anomaly,
                "observed_at": times[0] if times else None,
                "retrieved_at": now.isoformat(),
            }

            # Persist cache + manifest
            with cache_path.open("w", encoding="utf-8") as cf:
                json.dump(result, cf, indent=2)

            with manifest_path.open("w", encoding="utf-8") as mf:
                json.dump({
                    "source": self.name,
                    "dataset": self.dataset_name,
                    "retrieved_at": now.isoformat(),
                    "params": params,
                }, mf, indent=2)

            self._last_metadata = SourceMetadata(
                provider=self.name,
                dataset=self.dataset_name,
                source_url=self.FLOOD_API_URL,
                source_id="GLOFAS-LIVE",
                retrieved_at=now,
                license="CC-BY 4.0",
                freshness_seconds=0.0,
                status=ProviderStatus.LIVE,
                notes="Live modelled GloFAS catchment discharge.",
            )
            return result

        except Exception as exc:
            # Degraded or unavailable
            if cache_path.exists():
                try:
                    with cache_path.open("r", encoding="utf-8") as cf:
                        cached = json.load(cf)
                    cached["is_degraded"] = True
                    return cached
                except Exception:
                    pass

            return {
                "available": False,
                "data_type": "modelled_reanalysis_discharge",
                "is_physical_gauge": False,
                "latitude": lat,
                "longitude": lon,
                "river_discharge_m3s": None,
                "baseline_mean_m3s": None,
                "discharge_anomaly_ratio": None,
                "observed_at": None,
                "retrieved_at": now.isoformat(),
                "error": str(exc),
            }


hydrology_provider = HydrologyProvider()
