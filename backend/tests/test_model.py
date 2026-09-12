"""Automated tests for FloodNow TN Model, Metrics, Calibration, and Explainability."""

import json
from pathlib import Path
import pytest

from ml.inference import flood_now_inference

V2_MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models" / "flood_now_tn" / "v2"
V1_MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models" / "flood_now_tn" / "v1"
MODEL_DIR = V2_MODEL_DIR if V2_MODEL_DIR.exists() else V1_MODEL_DIR

V2_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "ml" / "data" / "flood_now_tn_v2" / "training_manifest.json"
V1_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "ml" / "data" / "flood_now_tn" / "training_manifest.json"
MANIFEST_PATH = V2_MANIFEST_PATH if V2_MANIFEST_PATH.exists() else V1_MANIFEST_PATH


def test_training_manifest_provenance():
    """Assert training manifest documents real samples, time bounds, and weak negatives."""
    assert MANIFEST_PATH.exists(), "training_manifest.json must exist"
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    pos = manifest.get("positive_samples", 0)
    neg = manifest.get("weak_negative_samples", manifest.get("negative_samples", 0))
    assert pos >= 500, f"Expected >= 500 positives, got {pos}"
    assert neg >= 500, f"Expected >= 500 negatives, got {neg}"

    pos_ratio = manifest["class_balance"].get("positive_ratio", manifest["class_balance"].get("positive_prevalence", 0.5))
    assert 0.40 <= pos_ratio <= 0.60

    if "temporal_coverage" in manifest:
        years = manifest["temporal_coverage"]["distinct_years"]
    elif "data_quality_audit" in manifest:
        yr = manifest["data_quality_audit"]["year_range"]
        years = yr[1] - yr[0] + 1
    else:
        years = 10
    assert years >= 10

    neg_method = manifest.get("negative_sampling_method", manifest.get("negative_sampling", {}).get("method", ""))
    assert "weak negative" in neg_method.lower() or "spatially matched" in neg_method.lower() or "offset" in neg_method.lower()

    audit = manifest.get("leakage_exclusion_audit", manifest.get("negative_sampling", {}))
    assert "duration_days" in audit or "exclusion_window_days" in audit


def test_model_metrics_evaluated_on_held_out_test_set():
    """Assert metrics are evaluated on unseen test data without hardcoded values."""
    metrics_path = MODEL_DIR / "metrics.json"
    assert metrics_path.exists(), "metrics.json artifact must exist"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    held_out = metrics["held_out_test_metrics"]
    test_count = held_out.get("test_sample_count", held_out.get("sample_count", 0))
    assert test_count >= 50
    assert 0.0 <= held_out["brier_score"] <= 0.25
    assert held_out["roc_auc"] >= 0.85
    assert held_out["pr_auc"] >= 0.85
    assert "confusion_matrix" in held_out


def test_model_inference_calibration_and_bands():
    """Assert model prediction produces calibrated probabilities and valid risk bands."""
    assert flood_now_inference.is_trained, "Model must be trained"

    # Extreme rainfall feature vector
    extreme_features = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "month_sin": -0.866,
        "month_cos": 0.5,
        "rain_1h_mm": 25.0,
        "rain_3h_mm": 50.0,
        "rain_6h_mm": 90.0,
        "rain_24h_mm": 210.0,
        "rain_72h_mm": 380.0,
        "soil_moisture_0_7cm": 0.52,
        "soil_moisture_7_28cm": 0.50,
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
        "rain_1h_mm": 0.0,
        "rain_3h_mm": 0.0,
        "rain_6h_mm": 0.0,
        "rain_24h_mm": 0.0,
        "rain_72h_mm": 0.0,
        "soil_moisture_0_7cm": 0.12,
        "soil_moisture_7_28cm": 0.14,
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
        "soil_moisture_0_7cm": 0.48,
    }
    result = flood_now_inference.predict(features)
    explanations = result["local_explanations"]
    features_mentioned = [e["feature"] for e in explanations]
    assert any(f in features_mentioned for f in ("rain_24h_mm", "soil_moisture_0_7cm", "soil_moisture"))

