from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

MODEL_DIR = Path(__file__).resolve().parent / "models" / "flood_now_tn" / "v1"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

FEATURE_NAMES = [
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "rain_24h_mm",
    "rain_72h_mm",
    "soil_moisture",
    "elevation_m",
    "slope_deg",
    "distance_to_water_km",
]


class ModelNotTrainedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=503,
            detail={
                "error": "MODEL_NOT_TRAINED",
                "message": (
                    "FloodNow TN model artifact is not trained yet. "
                    "Run 'python -m ml.trainer' to train and calibrate the model."
                ),
            },
        )


class FloodNowTNInferenceService:
    def __init__(self):
        self._model = None
        self._schema = {}
        self._metrics = {}
        self._importance = {}
        self._feature_means = {}
        self._load()

    def _load(self):
        if not MODEL_PATH.exists():
            return

        try:
            self._model = joblib.load(MODEL_PATH)
            if SCHEMA_PATH.exists():
                with SCHEMA_PATH.open("r", encoding="utf-8") as f:
                    self._schema = json.load(f)
                    self._feature_means = self._schema.get("feature_means", {})
            if METRICS_PATH.exists():
                with METRICS_PATH.open("r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            if IMPORTANCE_PATH.exists():
                with IMPORTANCE_PATH.open("r", encoding="utf-8") as f:
                    self._importance = json.load(f)
        except Exception:
            self._model = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "model_name": "FloodNow TN",
            "version": "1.0.0",
            "trained": self.is_trained,
            "status": "READY" if self.is_trained else "UNAVAILABLE",
            "algorithm": self._metrics.get("selected_algorithm", "None"),
            "held_out_metrics": self._metrics.get("held_out_test_metrics"),
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict calibrated flood occurrence probability and local explanations."""
        if not self.is_trained:
            raise ModelNotTrainedError()

        # Build feature dataframe
        input_data = {}
        for col in FEATURE_NAMES:
            val = features.get(col)
            if val is None:
                # Use dataset baseline mean
                val = self._feature_means.get(col, 0.0)
            input_data[col] = [float(val)]

        X = pd.DataFrame(input_data, columns=FEATURE_NAMES)
        prob = float(self._model.predict_proba(X)[0, 1])

        # Assign documented risk band
        if prob >= 0.80:
            band = "critical"
        elif prob >= 0.60:
            band = "high"
        elif prob >= 0.35:
            band = "moderate"
        else:
            band = "low"

        # Compute local per-prediction explanations (local feature contributions)
        local_explanations = self._compute_local_explanations(features)

        return {
            "flood_probability": round(prob, 4),
            "risk_band": band,
            "calibrated": True,
            "model_version": "FloodNow TN v1.0.0",
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "local_explanations": local_explanations,
        }

    def _compute_local_explanations(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        """Calculate local feature contributions compared to normal baseline."""
        contributions = []
        for col in FEATURE_NAMES:
            val = features.get(col)
            if val is None:
                continue
            mean_val = self._feature_means.get(col, 0.0)
            diff = float(val) - mean_val

            # High rainfall and soil moisture increase risk
            if col in ("rain_24h_mm", "rain_72h_mm", "soil_moisture") and diff > 0:
                impact = "increases_hazard"
                weight = round(diff / (mean_val if mean_val > 0 else 1.0), 2)
            elif col in ("elevation_m", "slope_deg") and diff < 0:
                impact = "increases_hazard_lowland"
                weight = round(abs(diff) / 100.0, 2)
            elif col == "distance_to_water_km" and diff < 0:
                impact = "near_river_channel"
                weight = round(abs(diff) / 5.0, 2)
            else:
                impact = "baseline_neutral"
                weight = 0.0

            if weight > 0.0:
                contributions.append({
                    "feature": col,
                    "observed_value": round(float(val), 2),
                    "baseline_mean": mean_val,
                    "impact": impact,
                    "attribution_weight": weight,
                })

        contributions.sort(key=lambda c: c["attribution_weight"], reverse=True)
        return contributions[:4]


flood_now_inference = FloodNowTNInferenceService()
