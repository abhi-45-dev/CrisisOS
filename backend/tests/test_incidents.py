import pytest


def test_report_citizen_incident(client):
    """Test reporting an unverified citizen incident in Tamil Nadu."""
    response = client.post(
        "/api/v1/incidents",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707,
            "incident_type": "waterlogging",
            "description": "Severe waterlogging on Mount Road near Thousand Lights.",
            "severity": "HIGH",
            "reporter_name": "R. Ramanathan",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("inc_")
    assert data["status"] == "UNVERIFIED"
    assert data["severity"] == "HIGH"
    assert data["latitude"] == 13.0827
    assert data["longitude"] == 80.2707
    assert data["reported_by"] == "R. Ramanathan"


def test_incident_xss_sanitization(client):
    """Test that HTML/script tags are strictly escaped in incident reports."""
    response = client.post(
        "/api/v1/incidents",
        json={
            "latitude": 13.0827,
            "longitude": 80.2707,
            "incident_type": "<script>alert('xss')</script>",
            "description": "Flooding <img src=x onerror=alert(1)> at bridge",
            "severity": "MEDIUM",
            "reporter_name": "<b>Hacker</b>",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "<script>" not in data["incident_type"]
    assert "&lt;script&gt;" in data["incident_type"]
    assert "<img" not in data["description"]
    assert "&lt;img" in data["description"]
    assert "<b>" not in data["reported_by"]
    assert "&lt;b&gt;" in data["reported_by"]


def test_incident_outside_tn_rejected(client):
    """Test that incident outside Tamil Nadu is rejected."""
    response = client.post(
        "/api/v1/incidents",
        json={
            "latitude": 19.0760,
            "longitude": 72.8777,  # Mumbai
            "incident_type": "waterlogging",
            "description": "Waterlogging in Mumbai",
            "severity": "LOW",
        },
    )
    assert response.status_code == 422


def test_list_and_nearby_incidents(client):
    """Test listing incidents and querying nearby incidents."""
    # List incidents
    list_resp = client.get("/api/v1/incidents")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert isinstance(items, list)

    # Nearby incidents
    nearby_resp = client.get("/api/v1/incidents/nearby?lat=13.0827&lon=80.2707&radius_km=50")
    assert nearby_resp.status_code == 200
    nearby_items = nearby_resp.json()
    assert isinstance(nearby_items, list)
    if nearby_items:
        assert "distance_km" in nearby_items[0]
