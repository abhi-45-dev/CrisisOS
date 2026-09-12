from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.providers.boundary_provider import boundary_provider
from app.providers.weather_provider import weather_provider
from ml.inference import flood_now_inference


class HazardService:
    """Statewide Tamil Nadu flood hazard surface generator and manager."""

    CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "cache" / "hazard" / "latest_hazard.json"

    def __init__(self):
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def get_statewide_hazard_grid(
        self,
        rainfall_override_mm: float | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Generate GeoJSON FeatureCollection of flood hazard cells covering Tamil Nadu."""
        # Use cached baseline unless force_refresh or scenario override
        if not force_refresh and rainfall_override_mm is None and self.CACHE_FILE.exists():
            try:
                with self.CACHE_FILE.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        cells = boundary_provider.generate_state_grid(step_deg=0.35)
        features = []
        now = datetime.now(timezone.utc)
        current_month = now.month

        # Batch infer across grid cells
        for cell in cells:
            lat = cell["latitude"]
            lon = cell["longitude"]

            # Query cached weather for coordinate
            weather = weather_provider.get_weather_at_coordinate(lat, lon)
            r24 = float(weather.get("rain_24h_mm") or 15.0)
            r72 = float(weather.get("rain_72h_mm") or 35.0)
            sm = float(weather.get("soil_moisture_0_7cm") or 0.28)

            if rainfall_override_mm is not None:
                r24 += rainfall_override_mm
                r72 += rainfall_override_mm * 1.5
                sm = min(0.65, sm + (rainfall_override_mm / 250.0))

            pred_features = {
                "latitude": lat,
                "longitude": lon,
                "month_sin": round(math_sin(current_month), 4),
                "month_cos": round(math_cos(current_month), 4),
                "rain_24h_mm": r24,
                "rain_72h_mm": r72,
                "soil_moisture": sm,
                "elevation_m": 45.0 if lon > 79.5 else 180.0,
                "slope_deg": 1.0 if lon > 79.5 else 3.5,
                "distance_to_water_km": 3.0 if lon > 79.8 else 10.0,
            }

            prediction = flood_now_inference.predict(pred_features)

            features.append({
                "type": "Feature",
                "id": cell["cell_id"],
                "properties": {
                    "cell_id": cell["cell_id"],
                    "flood_probability": prediction["flood_probability"],
                    "risk_band": prediction["risk_band"],
                    "observed_rain_24h_mm": r24,
                    "soil_moisture": sm,
                    "model_version": prediction["model_version"],
                    "inference_timestamp": prediction["inference_timestamp"],
                    "local_explanations": prediction["local_explanations"],
                },
                "geometry": cell["geometry"],
            })

        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "generated_at": now.isoformat(),
                "cell_count": len(features),
                "model_version": "FloodNow TN v1.0.0",
                "scenario_override_mm": rainfall_override_mm,
            },
            "features": features,
        }

        if rainfall_override_mm is None:
            with self.CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2)

        return geojson

    def predict_coordinate(self, lat: float, lon: float, rainfall_override_mm: float | None = None) -> dict[str, Any]:
        """Predict flood occurrence probability for an arbitrary coordinate inside Tamil Nadu."""
        boundary_provider.validate_coordinates(lat, lon)
        weather = weather_provider.get_weather_at_coordinate(lat, lon)

        now = datetime.now(timezone.utc)
        r24 = float(weather.get("rain_24h_mm") or 15.0)
        r72 = float(weather.get("rain_72h_mm") or 35.0)
        sm = float(weather.get("soil_moisture_0_7cm") or 0.28)

        if rainfall_override_mm is not None:
            r24 += rainfall_override_mm
            r72 += rainfall_override_mm * 1.5
            sm = min(0.65, sm + (rainfall_override_mm / 250.0))

        features = {
            "latitude": lat,
            "longitude": lon,
            "month_sin": round(math_sin(now.month), 4),
            "month_cos": round(math_cos(now.month), 4),
            "rain_24h_mm": r24,
            "rain_72h_mm": r72,
            "soil_moisture": sm,
            "elevation_m": 25.0 if lon > 79.5 else 120.0,
            "slope_deg": 1.2 if lon > 79.5 else 2.5,
            "distance_to_water_km": 2.5 if lon > 79.8 else 8.0,
        }

        res = flood_now_inference.predict(features)
        res["coordinates"] = [lat, lon]
        res["weather_inputs"] = {
            "rain_24h_mm": r24,
            "rain_72h_mm": r72,
            "soil_moisture": sm,
            "weather_source": weather.get("source", "Open-Meteo"),
            "observed_at": weather.get("observed_at"),
        }
        return res


def math_sin(month: int) -> float:
    import math
    return math.sin(2 * math.pi * month / 12.0)


def math_cos(month: int) -> float:
    import math
    return math.cos(2 * math.pi * month / 12.0)


hazard_service = HazardService()
