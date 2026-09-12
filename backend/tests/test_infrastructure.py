"""Automated tests for Tamil Nadu Infrastructure (Hospitals & Evacuation Facilities)."""

import pytest
from app.providers.infrastructure_provider import evacuation_facility_provider, hospital_provider


def test_hospitals_loaded_with_provenance():
    """Assert real hospitals are loaded with required attributes and no fake bed counts."""
    hospitals = hospital_provider.get_all()
    assert len(hospitals) >= 10, "Should load at least 10 key Tamil Nadu hospitals"

    for hosp in hospitals:
        assert hosp["id"].startswith("tn_hosp_")
        assert hosp["name"]
        assert 8.0 <= hosp["latitude"] <= 14.0
        assert 76.0 <= hosp["longitude"] <= 81.0
        # Truth in claims: available_beds and icu_capacity must be None
        assert hosp["available_beds"] is None, "Live available_beds must not be fabricated"
        assert hosp["icu_capacity"] is None, "Live icu_capacity must not be fabricated"


def test_shelters_loaded_with_classification():
    """Assert shelters are properly classified as official or candidate."""
    shelters = evacuation_facility_provider.get_all()
    assert len(shelters) >= 8, "Should load evacuation facilities across Tamil Nadu"

    official = [s for s in shelters if s.get("is_official")]
    candidate = [s for s in shelters if not s.get("is_official")]

    assert len(official) > 0, "Should have official relief shelters"
    assert len(candidate) > 0, "Should have candidate evacuation facilities"

    for c in candidate:
        assert c["classification"] == "CANDIDATE EVACUATION FACILITY"
        assert c["current_occupancy"] is None


def test_nearby_infrastructure_query(client):
    """Assert nearby infrastructure queries return facilities sorted by distance."""
    # Query near Chennai Central (13.0827, 80.2707)
    response = client.get("/api/v1/infrastructure/nearby?lat=13.0827&lon=80.2707&radius_km=15")
    assert response.status_code == 200
    data = response.json()

    assert len(data["hospitals"]) > 0
    assert len(data["shelters"]) > 0

    # Verify distance sorting
    dists = [h["distance_km"] for h in data["hospitals"]]
    assert dists == sorted(dists), "Hospitals must be sorted by distance"


def test_nearby_infrastructure_rejects_out_of_bounds(client):
    """Assert coordinates outside Tamil Nadu return HTTP 422."""
    response = client.get("/api/v1/infrastructure/nearby?lat=12.9716&lon=77.5946")  # Bengaluru
    assert response.status_code == 422
    assert "outside Tamil Nadu" in response.json()["detail"]
