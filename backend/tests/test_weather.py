"""Automated tests for coordinate-specific Weather and Hydrology Providers."""

import pytest
from app.providers.hydrology_provider import hydrology_provider
from app.providers.weather_provider import weather_provider


def test_weather_returns_none_not_zero_on_simulated_failure():
    """Verify scientific requirement: missing weather is NEVER converted to 0.0."""
    # Test parsing when payload contains empty data
    empty_payload = {"hourly": {"time": [], "precipitation": []}}
    parsed = weather_provider._parse_open_meteo_payload(
        empty_payload,
        lat=13.0827,
        lon=80.2707,
        retrieved_at=weather_provider.get_metadata().retrieved_at,
    )

    assert parsed["available"] is False
    assert parsed["rain_24h_mm"] is None, "Missing rain_24h_mm must be None, NOT 0.0"
    assert parsed["soil_moisture_0_7cm"] is None, "Missing soil_moisture must be None, NOT 0.0"


def test_weather_parsing_preserves_accumulations():
    """Verify correct multi-window accumulation from hourly values."""
    # Simulate 24 hours with 2.0 mm per hour
    times = [f"2026-09-12T{h:02d}:00" for h in range(24)]
    precip = [2.0] * 24
    sm0 = [0.25] * 24
    sm7 = [0.35] * 24

    payload = {
        "hourly": {
            "time": times,
            "precipitation": precip,
            "soil_moisture_0_to_7cm": sm0,
            "soil_moisture_7_to_28cm": sm7,
        }
    }

    parsed = weather_provider._parse_open_meteo_payload(
        payload,
        lat=13.0827,
        lon=80.2707,
        retrieved_at=weather_provider.get_metadata().retrieved_at,
    )

    assert parsed["available"] is True
    assert parsed["rain_1h_mm"] == 2.0
    assert parsed["rain_3h_mm"] == 6.0
    assert parsed["rain_6h_mm"] == 12.0
    assert parsed["rain_24h_mm"] == 48.0
    assert parsed["soil_moisture_0_7cm"] == 0.25


def test_distinct_weather_coordinates_for_chennai_and_coimbatore():
    """Verify that Chennai and Coimbatore are evaluated at distinct coordinates."""
    chennai_key = weather_provider._cache_key(13.0827, 80.2707)
    coimbatore_key = weather_provider._cache_key(11.0168, 76.9558)

    assert chennai_key != coimbatore_key, "Chennai and Coimbatore must have separate cache keys"
    assert "13.08_80.27" in str(chennai_key)
    assert "11.02_76.96" in str(coimbatore_key)


def test_hydrology_provider_modelled_discharge_semantics():
    """Verify that hydrology provider explicitly labels data as modelled, not gauge."""
    meta = hydrology_provider.get_metadata()
    assert meta.provider == "Copernicus GloFAS / Open-Meteo Flood API"
    assert "Modelled" in meta.dataset or "simulation" in meta.notes.lower()
