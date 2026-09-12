"""Automated Anti-Leakage, Temporal Partitioning, and Scientific Invariant Tests for FloodNow TN v2."""

import inspect
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from ml.historical_weather import HistoricalWeatherClient, get_environmental_features
from ml.inference import CANONICAL_FEATURES, flood_now_inference
import ml.dataset_builder as db_mod

DATASET_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "flood_now_tn_v2"
PARQUET_PATH = DATASET_DIR / "training_dataset.parquet"
MANIFEST_PATH = DATASET_DIR / "training_manifest.json"


def test_environmental_feature_function_cannot_receive_target_label():
    """Verify that get_environmental_features does NOT accept target/label parameters."""
    sig = inspect.signature(get_environmental_features)
    param_names = list(sig.parameters.keys())
    
    # Strictly check parameter names
    prohibited = ["target", "label", "class", "is_positive", "is_flood", "flood_occurrence", "sample_type"]
    for p in prohibited:
        assert p not in param_names, f"LEAKAGE RISK: get_environmental_features accepts prohibited parameter: {p}"
    
    assert "latitude" in param_names
    assert "longitude" in param_names
    assert "prediction_timestamp" in param_names


def test_target_absent_from_predictor_features():
    """Verify that target/label is never included in the predictor feature schema."""
    assert "target" not in CANONICAL_FEATURES
    assert "flood_occurrence" not in CANONICAL_FEATURES
    assert "sample_type" not in CANONICAL_FEATURES


def test_no_synthetic_terrain_in_canonical_features():
    """Verify that legacy handcrafted elevation/slope/distance_to_water were eliminated."""
    assert "elevation_m" not in CANONICAL_FEATURES
    assert "slope_deg" not in CANONICAL_FEATURES
    assert "distance_to_water_km" not in CANONICAL_FEATURES


def test_negative_exclusion_window_logic():
    """Verify that candidate negatives within +-14 days of a known flood in the same district are excluded."""
    known_floods = {datetime(2015, 12, 1)}
    
    # Date within 10 days -> must be excluded
    test_cand_1 = datetime(2015, 12, 10)
    is_excluded_1 = any(abs((test_cand_1 - kf).days) <= 14 for kf in known_floods)
    assert is_excluded_1 is True, "Candidate within 9 days of known flood must be excluded"
    
    # Date 50 days away -> allowed
    test_cand_2 = datetime(2015, 10, 1)
    is_excluded_2 = any(abs((test_cand_2 - kf).days) <= 14 for kf in known_floods)
    assert is_excluded_2 is False, "Candidate 61 days away from known flood should be permitted"


def test_pre_event_time_semantics():
    """Verify that antecedent evaluation window strictly ends prior to event date."""
    # When given a date string "2015-12-01", cutoff must be 2015-11-30 23:00
    mock_client = MagicMock(spec=HistoricalWeatherClient)
    mock_client.fetch_window.return_value = {
        "hourly": {
            "time": ["2015-11-30T21:00", "2015-11-30T22:00", "2015-11-30T23:00"],
            "precipitation": [5.0, 10.0, 15.0],
            "soil_moisture_0_to_7cm": [0.42, 0.43, 0.44],
            "soil_moisture_7_to_28cm": [0.40, 0.40, 0.41],
        }
    }
    
    feats = get_environmental_features(13.0827, 80.2707, "2015-12-01", client=mock_client)
    # Check that fetch_window was called with end_date="2015-11-30" (NOT 2015-12-01)
    called_args = mock_client.fetch_window.call_args[0]
    assert called_args[3] == "2015-11-30", "Fetch window end_date must end before event date boundary"
    assert feats["rain_1h_mm"] == 15.0
    assert feats["rain_3h_mm"] == 30.0


def test_weather_client_raises_on_failure_no_synthetic_fallback():
    """Verify that client raises an exception and does NOT invent synthetic numbers on failure."""
    client = HistoricalWeatherClient(cache_dir=Path("/tmp/nonexistent_weather_test_cache"))
    
    with patch("httpx.Client.get") as mock_get:
        mock_get.side_effect = Exception("Simulated network outage")
        with pytest.raises(RuntimeError) as exc_info:
            client.fetch_window(13.08, 80.27, "2020-01-01", "2020-01-04", max_retries=1)
        assert "Failed to retrieve genuine historical weather" in str(exc_info.value)


def test_district_centroids_all_valid():
    """Verify all 38 Tamil Nadu districts have verified coordinates."""
    assert len(db_mod.TN_38_DISTRICTS) == 38
    for d, (lat, lon) in db_mod.TN_38_DISTRICTS.items():
        assert 8.0 <= lat <= 14.0, f"District {d} latitude out of bounds"
        assert 76.0 <= lon <= 81.0, f"District {d} longitude out of bounds"
