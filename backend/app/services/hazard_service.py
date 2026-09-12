from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.providers.boundary_provider import boundary_provider
from app.providers.weather_provider import weather_provider
from ml.inference import flood_now_inference


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


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

            # Query cached weather for coordinate without synthetic fallback fabrication
            weather = weather_provider.get_weather_at_coordinate(lat, lon)
            obs_r1 = _safe_float(weather.get("rain_1h_mm"))
            obs_r3 = _safe_float(weather.get("rain_3h_mm"))
            obs_r6 = _safe_float(weather.get("rain_6h_mm"))
            obs_r24 = _safe_float(weather.get("rain_24h_mm"))
            obs_r72 = _safe_float(weather.get("rain_72h_mm"))
            obs_sm_top = _safe_float(weather.get("soil_moisture_0_7cm"))
            obs_sm_root = _safe_float(weather.get("soil_moisture_7_28cm"))
            if obs_sm_root is None and obs_sm_top is not None:
                obs_sm_root = obs_sm_top

            r1 = obs_r1
            r3 = obs_r3
            r6 = obs_r6
            r24 = obs_r24
            r72 = obs_r72
            sm_top = obs_sm_top
            sm_root = obs_sm_root

            if rainfall_override_mm is not None:
                base_r24 = r24 if r24 is not None else 0.0
                base_r72 = r72 if r72 is not None else 0.0
                r1 = (r1 or 0.0) + rainfall_override_mm * 0.1
                r3 = (r3 or 0.0) + rainfall_override_mm * 0.25
                r6 = (r6 or 0.0) + rainfall_override_mm * 0.5
                r24 = base_r24 + rainfall_override_mm
                r72 = base_r72 + rainfall_override_mm * 1.5
                if sm_top is not None:
                    sm_top = min(0.65, sm_top + (rainfall_override_mm / 250.0))
                if sm_root is not None:
                    sm_root = min(0.65, sm_root + (rainfall_override_mm / 300.0))

            pred_features = {
                "latitude": lat,
                "longitude": lon,
                "month_sin": round(math_sin(current_month), 4),
                "month_cos": round(math_cos(current_month), 4),
                "rain_1h_mm": r1,
                "rain_3h_mm": r3,
                "rain_6h_mm": r6,
                "rain_24h_mm": r24,
                "rain_72h_mm": r72,
                "soil_moisture_0_7cm": sm_top,
                "soil_moisture_7_28cm": sm_root,
            }

            prediction = flood_now_inference.predict(pred_features)

            features.append({
                "type": "Feature",
                "id": cell["cell_id"],
                "properties": {
                    "cell_id": cell["cell_id"],
                    "flood_probability": prediction["flood_probability"],
                    "risk_band": prediction["risk_band"],
                    "observed_rain_24h_mm": obs_r24,
                    "soil_moisture": obs_sm_top,
                    "missing_inputs": prediction.get("missing_inputs", []),
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
                "model_version": "FloodNow TN v2.0.0",
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
        obs_r1 = _safe_float(weather.get("rain_1h_mm"))
        obs_r3 = _safe_float(weather.get("rain_3h_mm"))
        obs_r6 = _safe_float(weather.get("rain_6h_mm"))
        obs_r24 = _safe_float(weather.get("rain_24h_mm"))
        obs_r72 = _safe_float(weather.get("rain_72h_mm"))
        obs_sm_top = _safe_float(weather.get("soil_moisture_0_7cm"))
        obs_sm_root = _safe_float(weather.get("soil_moisture_7_28cm"))
        if obs_sm_root is None and obs_sm_top is not None:
            obs_sm_root = obs_sm_top

        r1 = obs_r1
        r3 = obs_r3
        r6 = obs_r6
        r24 = obs_r24
        r72 = obs_r72
        sm_top = obs_sm_top
        sm_root = obs_sm_root

        baseline_type = "observed"
        if rainfall_override_mm is not None:
            if r24 is None:
                baseline_type = "dry_baseline_assumed (missing_observed_weather)"
            base_r24 = r24 if r24 is not None else 0.0
            base_r72 = r72 if r72 is not None else 0.0
            r1 = (r1 or 0.0) + rainfall_override_mm * 0.1
            r3 = (r3 or 0.0) + rainfall_override_mm * 0.25
            r6 = (r6 or 0.0) + rainfall_override_mm * 0.5
            r24 = base_r24 + rainfall_override_mm
            r72 = base_r72 + rainfall_override_mm * 1.5
            if sm_top is not None:
                sm_top = min(0.65, sm_top + (rainfall_override_mm / 250.0))
            if sm_root is not None:
                sm_root = min(0.65, sm_root + (rainfall_override_mm / 300.0))

        features = {
            "latitude": lat,
            "longitude": lon,
            "month_sin": round(math_sin(now.month), 4),
            "month_cos": round(math_cos(now.month), 4),
            "rain_1h_mm": r1,
            "rain_3h_mm": r3,
            "rain_6h_mm": r6,
            "rain_24h_mm": r24,
            "rain_72h_mm": r72,
            "soil_moisture_0_7cm": sm_top,
            "soil_moisture_7_28cm": sm_root,
        }

        res = flood_now_inference.predict(features)
        res["coordinates"] = [lat, lon]
        res["weather_inputs"] = {
            "observed_rain_1h_mm": obs_r1,
            "observed_rain_3h_mm": obs_r3,
            "observed_rain_6h_mm": obs_r6,
            "observed_rain_24h_mm": obs_r24,
            "observed_rain_72h_mm": obs_r72,
            "observed_soil_moisture_0_7cm": obs_sm_top,
            "observed_soil_moisture_7_28cm": obs_sm_root,
            "rain_24h_mm": r24,
            "rain_72h_mm": r72,
            "soil_moisture": sm_top,
            "soil_moisture_0_7cm": sm_top,
            "soil_moisture_7_28cm": sm_root,
            "weather_source": weather.get("source", "Open-Meteo"),
            "observed_at": weather.get("observed_at"),
            "baseline_type": baseline_type,
            "missing_observations": [
                k for k, v in [
                    ("rain_1h_mm", obs_r1),
                    ("rain_3h_mm", obs_r3),
                    ("rain_6h_mm", obs_r6),
                    ("rain_24h_mm", obs_r24),
                    ("rain_72h_mm", obs_r72),
                    ("soil_moisture_0_7cm", obs_sm_top),
                    ("soil_moisture_7_28cm", obs_sm_root),
                ] if v is None
            ],
            "missing_inputs": res.get("missing_inputs", []),
        }
        return res


def math_sin(month: int) -> float:
    import math
    return math.sin(2 * math.pi * month / 12.0)


def math_cos(month: int) -> float:
    import math
    return math.cos(2 * math.pi * month / 12.0)


hazard_service = HazardService()
