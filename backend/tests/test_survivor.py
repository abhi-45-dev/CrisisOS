import pytest


def test_survivor_recommendation_chennai(client):
    """Test survivor evacuation recommendation in Chennai."""
    response = client.post(
        "/api/v1/survivor/recommend",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707,
            "preferred_type": "all",
            "language": "en",
            "radius_km": 30.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "survivor_location" in data
    assert "survivor_hazard" in data
    assert "verdict" in data
    assert data["verdict"] in ("RECOMMENDED", "NO ROUTE CONFIDENTLY RECOMMENDED", "NO_FACILITY_WITHIN_RANGE")
    assert "rationale" in data
    assert len(data["rationale"]) > 10

    if data["verdict"] == "RECOMMENDED":
        dest = data["recommended_destination"]
        assert dest is not None
        assert "name" in dest
        assert "latitude" in dest
        assert "longitude" in dest
        assert "recommended_route" in data
        assert "fastest_alternative_route" in data
        # Road routing metrics
        assert "total_distance_km" in data["recommended_route"]
        assert "risk_verdict" in data["recommended_route"]
        assert "hazard_segments" in data["recommended_route"]


def test_survivor_tamil_localization(client):
    """Test Tamil language localization in survivor recommendation."""
    response = client.post(
        "/api/v1/survivor/recommend",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707,
            "preferred_type": "shelter",
            "language": "ta",
            "radius_km": 35.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rationale_ta"] is not None
    # Tamil Unicode block check
    assert any("\u0b80" <= ch <= "\u0bff" for ch in data["rationale_ta"])


def test_survivor_out_of_bounds_rejected(client):
    """Test that coordinates outside Tamil Nadu are rejected with 422."""
    response = client.post(
        "/api/v1/survivor/recommend",
        json={
            "latitude": 28.6139,
            "longitude": 77.2090,  # New Delhi
            "preferred_type": "all",
            "language": "en",
        },
    )
    assert response.status_code == 422
    assert "outside Tamil Nadu" in response.json()["detail"]


def test_survivor_hospital_filter(client):
    """Test filtering survivor recommendation strictly to hospitals."""
    response = client.post(
        "/api/v1/survivor/recommend",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707,
            "preferred_type": "hospital",
            "language": "en",
            "radius_km": 25.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    if data["verdict"] == "RECOMMENDED":
        dest = data["recommended_destination"]
        assert dest.get("facility_type") == "hospital" or dest.get("category") == "hospital"
