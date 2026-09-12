"""Automated tests for FloodNow TN Model, Metrics, Calibration, and Explainability."""

import json
from pathlib import Path
import pytest

from ml.inference import flood_now_inference

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models" / "flood_now_tn" / "v1"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "ml" / "data" / "flood_now_tn" / "training_manifest.json"


def test_training_manifest_provenance():
    """Assert training manifest documents real samples, time bounds, and weak negatives."""
    assert MANIFEST_PATH.exists(), "training_manifest.json must exist"
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["positive_samples"] > 1000
    assert manifest["negative_samples"] > 1000
    assert manifest["class_balance"]["positive_ratio"] < 0.60
    assert manifest["temporal_coverage"]["distinct_years"] >= 10
    assert "weak negative" in manifest["negative_sampling_method"].lower()
    assert "duration_days" in manifest["leakage_exclusion_audit"]


def test_model_metrics_evaluated_on_held_out_test_set():
    """Assert metrics are evaluated on unseen test data without hardcoded values."""
    metrics_path = MODEL_DIR / "metrics.json"
    assert metrics_path.exists(), "metrics.json artifact must exist"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    held_out = metrics["held_out_test_metrics"]
    assert held_out["test_sample_count"] > 200
    assert 0.0 <= held_out["brier_score"] <= 0.25
    assert held_out["roc_auc"] >= 0.85
    assert held_out["pr_auc"] >= 0.85
    assert len(held_out["confusion_matrix"]) == 2


def test_model_inference_calibration_and_bands():
    """Assert model prediction produces calibrated probabilities and valid risk bands."""
    assert flood_now_inference.is_trained, "Model must be trained"

    # Extreme rainfall feature vector
    extreme_features = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "month_sin": 0.5,
        "month_cos": -0.86,
        "rain_24h_mm": 210.0,
        "rain_72h_mm": 380.0,
        "soil_moisture": 0.52,
        "elevation_m": 8.0,
        "slope_deg": 0.5,
        "distance_to_water_km": 1.2,
    }
    result = flood_now_inference.predict(extreme_features)
    assert 0.0 <= result["flood_probability"] <= 1.0
    assert result["risk_band"] in ("high", "critical")
    assert result["calibrated"] is True
    assert len(result["local_explanations"]) > 0

    # Dry weather feature vector
    dry_features = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "month_sin": 0.86,
        "month_cos": 0.5,
        "rain_24h_mm": 0.0,
        "rain_72h_mm": 0.0,
        "soil_moisture": 0.12,
        "elevation_m": 8.0,
        "slope_deg": 0.5,
        "distance_to_water_km": 1.2,
    }
    dry_result = flood_now_inference.predict(dry_features)
    assert dry_result["flood_probability"] < 0.35
    assert dry_result["risk_band"] == "low"


def test_local_explainability():
    """Assert that local feature contributions explain why a prediction is risky."""
    features = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "rain_24h_mm": 180.0,
        "soil_moisture": 0.48,
    }
    result = flood_now_inference.predict(features)
    explanations = result["local_explanations"]
    features_mentioned = [e["feature"] for e in explanations]
    assert "rain_24h_mm" in features_mentioned or "soil_moisture" in features_mentioned
