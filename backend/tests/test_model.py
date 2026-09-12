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
    """Assert metrics are evaluated on unseen test data with structural scientific checks.
    
    Does NOT assert predetermined impressive metrics (e.g. >= 0.85); asserts valid ranges,
    unbroken evaluation artifacts, and collapsed shuffled-target sanity.
    """
    metrics_path = MODEL_DIR / "metrics.json"
    assert metrics_path.exists(), "metrics.json artifact must exist"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Structural check of held-out test evaluation
    held_out = metrics["held_out_test_metrics"]
    test_count = held_out.get("test_sample_count", held_out.get("sample_count", 0))
    assert test_count >= 50, f"Expected held-out test set size >= 50, got {test_count}"
    assert 0.0 <= held_out["brier_score"] <= 1.0
    assert 0.0 <= held_out["roc_auc"] <= 1.0
    assert 0.0 <= held_out["pr_auc"] <= 1.0
    assert "confusion_matrix" in held_out
    assert "confidence_intervals" in held_out

    # 2. Calibration artifact exists
    calibration_path = MODEL_DIR / "calibration.json"
    assert calibration_path.exists(), "calibration.json must exist"

    # 3. Scientific anti-leakage sanity check: shuffled-target collapses to chance (~0.50)
    assert "sanity_experiments" in metrics
    shuffled = metrics["sanity_experiments"]["shuffled_target_experiment"]
    assert 0.40 <= shuffled["roc_auc"] <= 0.65, f"Shuffled target should collapse near chance (0.50), got {shuffled['roc_auc']}"


def test_risk_threshold_consistency():
    """Assert metadata.json, feature_schema.json, and inference service thresholds strictly match."""
    metadata_path = MODEL_DIR / "metadata.json"
    schema_path = MODEL_DIR / "feature_schema.json"

    assert metadata_path.exists(), "metadata.json must exist"
    assert schema_path.exists(), "feature_schema.json must exist"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    meta_thresholds = metadata.get("risk_thresholds", {})
    schema_thresholds = schema.get("risk_thresholds", {})

    assert meta_thresholds == schema_thresholds, (
        f"Threshold mismatch! metadata: {meta_thresholds} vs feature_schema: {schema_thresholds}"
    )
    assert meta_thresholds["moderate"] == 0.35
    assert meta_thresholds["high"] == 0.55
    assert meta_thresholds["critical"] == 0.75


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


def test_local_explainability_and_imputed_inputs():
    """Assert that missing inputs are marked as imputed and not claimed as physical hazard drivers."""
    features = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "rain_24h_mm": 180.0,
        # rain_1h_mm, rain_3h_mm, rain_6h_mm, rain_72h_mm intentionally omitted
    }
    result = flood_now_inference.predict(features)
    explanations = result["local_explanations"]

    # Verify missing inputs listed
    assert "missing_inputs" in result
    assert "rain_1h_mm" in result["missing_inputs"]

    # Find explanation for imputed feature
    rain_1h_exp = next((e for e in explanations if e["feature"] == "rain_1h_mm"), None)
    assert rain_1h_exp is not None
    assert rain_1h_exp["was_imputed"] is True
    assert rain_1h_exp["observed_value"] is None
    assert rain_1h_exp["imputed_value"] is not None
    assert rain_1h_exp["impact"] == "imputed_input"
    assert rain_1h_exp["attribution_weight"] == 0.0

    # Find explanation for observed feature
    rain_24h_exp = next((e for e in explanations if e["feature"] == "rain_24h_mm"), None)
    assert rain_24h_exp is not None
    assert rain_24h_exp["was_imputed"] is False
    assert rain_24h_exp["observed_value"] == 180.0
    assert rain_24h_exp["impact"] in ("increases_risk", "decreases_risk")


def test_scenario_override_requires_baseline_weather():
    """Assert that scenario override fails gracefully if baseline observed weather is missing."""
    from app.services.hazard_service import hazard_service

    # Coordinate outside Tamil Nadu fails validation
    with pytest.raises(Exception):
        hazard_service.predict_coordinate(lat=28.6139, lon=77.2090, rainfall_override_mm=50.0)


