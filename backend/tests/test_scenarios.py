import pytest


def test_rainfall_scenario_simulation(client):
    """Test what-if rainfall delta simulation through FloodNow TN model."""
    response = client.post(
        "/api/v1/scenarios/rainfall",
        json={
            "rainfall_delta_mm": 50.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "scenario" in data
    assert data["scenario"]["rainfall_delta_mm"] == 50.0
    assert "FloodNow TN" in data["scenario"]["model_version"]
    assert "summary" in data["scenario"]
    assert "metrics" in data["scenario"]
    metrics = data["scenario"]["metrics"]
    assert metrics["total_cells_evaluated"] > 50
    assert "average_probability_delta" in metrics

    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    features = data["geojson"]["features"]
    assert len(features) > 50

    # Ensure each feature has both baseline and scenario probabilities
    first_props = features[0]["properties"]
    assert "baseline_flood_probability" in first_props
    assert "scenario_flood_probability" in first_props
    assert "delta_flood_probability" in first_props
    assert first_props["scenario_flood_probability"] >= first_props["baseline_flood_probability"]


def test_rainfall_scenario_invalid_delta(client):
    """Test validation reject for negative or absurd rainfall delta."""
    response = client.post(
        "/api/v1/scenarios/rainfall",
        json={
            "rainfall_delta_mm": -10.0,
        },
    )
    assert response.status_code == 422
