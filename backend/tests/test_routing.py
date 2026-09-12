"""Automated tests for Real Road Routing and Risk-Aware Route Comparison."""

import pytest
from app.routing.provider import RoutingProfile, osm_routing_provider


def test_osm_routing_chennai_corridor():
    """Assert routing calculates valid path on real OSM road network."""
    # Chennai Central (13.0827, 80.2707) to Tambaram Sanatorium (12.9249, 80.1265)
    result = osm_routing_provider.calculate_route(
        13.0827, 80.2707,
        12.9249, 80.1265,
        profile=RoutingProfile.FASTEST,
    )
    assert result["feasible"] is True
    assert result["total_distance_km"] > 15.0
    assert result["estimated_duration_min"] > 10.0
    assert len(result["coordinates"]) >= 4
    assert result["geometry"]["type"] == "LineString"
    assert "Grand Southern Trunk (GST) Road" in " ".join(result["road_names"])


def test_osm_routing_interdistrict():
    """Assert long-distance routing connects Chennai to Coimbatore."""
    # Chennai to Coimbatore
    result = osm_routing_provider.calculate_route(
        13.0827, 80.2707,  # Chennai
        11.0168, 76.9678,  # Coimbatore
        profile=RoutingProfile.FASTEST,
    )
    assert result["feasible"] is True
    assert result["total_distance_km"] > 350.0
    assert "Salem" in " ".join(result["road_names"]) or "NH 544" in " ".join(result["road_names"])


def test_hazard_polygons_alter_safest_routing(client):
    """Test that introducing a flood hazard polygon causes SAFEST route to avoid the hazard."""
    # Anna Nagar (13.0855, 80.2154) to Tambaram (12.9249, 80.1265)
    # Direct path is Inner Ring Rd -> Guindy Kathipara -> GST Rd.
    # We place a critical flood hazard polygon over Guindy Kathipara (13.0067, 80.2025).
    guindy_hazard_poly = {
        "cell_id": "hazard_guindy_flood",
        "flood_probability": 0.92,
        "risk_band": "critical",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [80.1900, 12.9950],
                [80.2150, 12.9950],
                [80.2150, 13.0200],
                [80.1900, 13.0200],
                [80.1900, 12.9950],
            ]],
        },
    }

    payload = {
        "origin": {"latitude": 13.0855, "longitude": 80.2154},
        "destination": {"latitude": 12.9249, "longitude": 80.1265},
        "hazard_polygons": [guindy_hazard_poly],
    }

    response = client.post("/api/v1/routes/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    fastest = data["routes"]["fastest"]
    safest = data["routes"]["safest"]

    # FASTEST chooses the shortest time path and hits Guindy Kathipara flood zone
    assert fastest["max_flood_probability"] >= 0.80, "FASTEST route should cross the hazard"
    assert fastest["critical_risk_distance_km"] > 0.0

    # SAFEST route penalizes or detours around Guindy
    assert safest["critical_risk_distance_km"] < fastest["critical_risk_distance_km"] or (
        safest["mean_flood_probability"] < fastest["mean_flood_probability"]
    )
    assert data["recommendation"]["selected_profile"] == "safest"
    assert "SAFEST ROUTE" in data["recommendation"]["verdict"]


def test_routes_compare_rejects_out_of_bounds(client):
    """Assert that origin outside Tamil Nadu returns HTTP 422."""
    payload = {
        "origin": {"latitude": 12.9716, "longitude": 77.5946},  # Bengaluru
        "destination": {"latitude": 13.0827, "longitude": 80.2707},  # Chennai
    }
    response = client.post("/api/v1/routes/compare", json=payload)
    assert response.status_code == 422
    assert "outside Tamil Nadu" in response.json()["detail"]
