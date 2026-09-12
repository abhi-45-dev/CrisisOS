from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.providers.boundary_provider import boundary_provider
from app.providers.infrastructure_provider import evacuation_facility_provider, hospital_provider

router = APIRouter(prefix="/infrastructure")


@router.get("/nearby")
def get_nearby_infrastructure(
    lat: float = Query(..., description="Latitude inside Tamil Nadu"),
    lon: float = Query(..., description="Longitude inside Tamil Nadu"),
    radius_km: float = Query(default=25.0, ge=1.0, le=150.0),
    facility_type: Literal["all", "hospital", "shelter"] = Query(default="all"),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Find real hospitals and evacuation facilities near an arbitrary Tamil Nadu coordinate."""
    # 1. Strict coordinate validation against Tamil Nadu boundary
    boundary_provider.validate_coordinates(lat, lon)

    results = {
        "query": {"latitude": lat, "longitude": lon, "radius_km": radius_km},
        "hospitals": [],
        "shelters": [],
    }

    if facility_type in ("all", "hospital"):
        results["hospitals"] = hospital_provider.get_nearby(lat, lon, radius_km=radius_km, limit=limit)

    if facility_type in ("all", "shelter"):
        results["shelters"] = evacuation_facility_provider.get_nearby(lat, lon, radius_km=radius_km, limit=limit)

    return results


@router.get("/hospitals")
def list_hospitals(district: str | None = Query(default=None)):
    """List genuine geocoded Tamil Nadu hospitals."""
    records = hospital_provider.get_all(district=district)
    return {
        "count": len(records),
        "source": hospital_provider.name,
        "live_beds_available": False,
        "hospitals": records,
    }


@router.get("/shelters")
def list_shelters(official_only: bool = Query(default=False)):
    """List genuine Tamil Nadu evacuation shelters and candidate relief centres."""
    records = evacuation_facility_provider.get_all(is_official_only=official_only)
    return {
        "count": len(records),
        "source": evacuation_facility_provider.name,
        "shelters": records,
    }
