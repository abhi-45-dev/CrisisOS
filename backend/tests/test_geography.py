"""Automated tests for Tamil Nadu geographic boundary and coordinate validation."""

import pytest
from fastapi import HTTPException

from app.providers.boundary_provider import boundary_provider

# Known coordinates inside Tamil Nadu
TAMIL_NADU_LOCATIONS = [
    ("Chennai Central", 13.0827, 80.2707),
    ("Coimbatore", 11.0168, 76.9558),
    ("Madurai Meenakshi", 9.9195, 78.1193),
    ("Tiruchirappalli", 10.7905, 78.7047),
    ("Salem", 11.6643, 78.1460),
    ("Tirunelveli", 8.7139, 77.7567),
    ("Nagercoil", 8.1833, 77.4119),
    ("Kanyakumari", 8.0883, 77.5385),
    ("Thanjavur", 10.7870, 79.1378),
    ("Vellore", 12.9165, 79.1325),
    ("Erode", 11.3410, 77.7172),
]

# Known coordinates OUTSIDE Tamil Nadu
OUT_OF_BOUNDS_LOCATIONS = [
    ("Bengaluru (Karnataka)", 12.9716, 77.5946),
    ("Kochi (Kerala)", 9.9312, 76.2673),
    ("Tirupati (Andhra Pradesh)", 13.6288, 79.4192),
    ("Mumbai (Maharashtra)", 19.0760, 72.8777),
    ("New Delhi", 28.6139, 77.2090),
    ("Bay of Bengal Ocean", 12.5000, 82.5000),
    ("Arabian Sea Ocean", 9.0000, 74.5000),
]


@pytest.mark.parametrize("name,lat,lon", TAMIL_NADU_LOCATIONS)
def test_valid_tamil_nadu_coordinates(name: str, lat: float, lon: float):
    """Assert that genuine Tamil Nadu coordinates evaluate to inside state."""
    assert boundary_provider.is_inside_tamil_nadu(lat, lon), f"{name} ({lat}, {lon}) should be inside TN"


@pytest.mark.parametrize("name,lat,lon", OUT_OF_BOUNDS_LOCATIONS)
def test_out_of_bounds_coordinates(name: str, lat: float, lon: float):
    """Assert that coordinates outside Tamil Nadu evaluate to false."""
    assert not boundary_provider.is_inside_tamil_nadu(lat, lon), f"{name} ({lat}, {lon}) should be OUTSIDE TN"


def test_validate_coordinates_raises_for_out_of_bounds():
    """Assert that validate_coordinates raises HTTP 422 for outside points."""
    with pytest.raises(HTTPException) as exc_info:
        boundary_provider.validate_coordinates(12.9716, 77.5946)  # Bengaluru
    assert exc_info.value.status_code == 422
    assert "outside Tamil Nadu" in exc_info.value.detail


def test_validate_coordinates_passes_for_chennai():
    """Assert that validate_coordinates completes cleanly for Chennai."""
    boundary_provider.validate_coordinates(13.0827, 80.2707)


def test_generate_state_grid():
    """Assert statewide grid generates valid cells enclosed in Tamil Nadu."""
    cells = boundary_provider.generate_state_grid(step_deg=0.5)
    assert len(cells) > 20, "Should generate multiple grid cells across Tamil Nadu"
    for cell in cells:
        assert cell["cell_id"].startswith("tn_grid_")
        assert boundary_provider.is_inside_tamil_nadu(cell["latitude"], cell["longitude"])
        assert cell["geometry"]["type"] == "Polygon"
