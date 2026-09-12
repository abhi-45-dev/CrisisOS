from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from fastapi import APIRouter

from app.providers.boundary_provider import boundary_provider
from app.routing.exposure import analyze_route_exposure
from app.routing.provider import RoutingProfile, osm_routing_provider

router = APIRouter(prefix="/routes")


class GeoCoordinate(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)


class RouteCompareRequest(BaseModel):
    origin: GeoCoordinate
    destination: GeoCoordinate
    hazard_polygons: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/compare")
def compare_routes(body: RouteCompareRequest):
    """Calculate and compare FASTEST, BALANCED, and SAFEST routes over real OSM roads.

    Both coordinates must be inside Tamil Nadu.
    Routes are independently evaluated against the ML flood hazard surface.
    """
    # 1. Strict coordinate validation
    boundary_provider.validate_coordinates(body.origin.latitude, body.origin.longitude)
    boundary_provider.validate_coordinates(body.destination.latitude, body.destination.longitude)

    # 2. Compute candidate paths over real OSM road network
    fastest_raw = osm_routing_provider.calculate_route(
        body.origin.latitude,
        body.origin.longitude,
        body.destination.latitude,
        body.destination.longitude,
        profile=RoutingProfile.FASTEST,
        hazard_polygons=body.hazard_polygons,
    )
    balanced_raw = osm_routing_provider.calculate_route(
        body.origin.latitude,
        body.origin.longitude,
        body.destination.latitude,
        body.destination.longitude,
        profile=RoutingProfile.BALANCED,
        hazard_polygons=body.hazard_polygons,
    )
    safest_raw = osm_routing_provider.calculate_route(
        body.origin.latitude,
        body.origin.longitude,
        body.destination.latitude,
        body.destination.longitude,
        profile=RoutingProfile.SAFEST,
        hazard_polygons=body.hazard_polygons,
    )

    # 3. Independent flood exposure analysis
    fastest = analyze_route_exposure(fastest_raw, body.hazard_polygons)
    balanced = analyze_route_exposure(balanced_raw, body.hazard_polygons)
    safest = analyze_route_exposure(safest_raw, body.hazard_polygons)

    # 4. Generate objective, explainable comparison
    recommendation = _build_recommendation(fastest, balanced, safest)

    return {
        "origin": [body.origin.latitude, body.origin.longitude],
        "destination": [body.destination.latitude, body.destination.longitude],
        "routes": {
            "fastest": fastest,
            "balanced": balanced,
            "safest": safest,
        },
        "recommendation": recommendation,
    }


def _build_recommendation(fastest: dict, balanced: dict, safest: dict) -> dict[str, Any]:
    if not fastest.get("feasible"):
        return {
            "selected_profile": "none",
            "verdict": "NO ROUTE CONFIDENTLY RECOMMENDED",
            "rationale": "No reachable path found connecting origin and destination on the road network.",
        }

    f_crit = fastest.get("critical_risk_distance_km", 0.0)
    s_crit = safest.get("critical_risk_distance_km", 0.0)
    f_high = fastest.get("high_risk_distance_km", 0.0)
    s_high = safest.get("high_risk_distance_km", 0.0)

    time_diff = round(safest.get("baseline_network_duration_min", 0.0) - fastest.get("baseline_network_duration_min", 0.0), 1)
    dist_diff = round(safest.get("total_distance_km", 0.0) - fastest.get("total_distance_km", 0.0), 1)

    if f_crit > 0.0 and s_crit < f_crit:
        return {
            "selected_profile": "safest",
            "verdict": "SAFEST ROUTE STRONGLY RECOMMENDED",
            "rationale": (
                f"The FASTEST route intersects {f_crit:.1f} km of predicted CRITICAL flood exposure. "
                f"The SAFEST route diverts (+{time_diff} min, +{dist_diff} km) to reduce critical flood exposure to {s_crit:.1f} km."
            ),
        }
    elif (f_high > 0.5 and s_high < f_high) or (safest.get("mean_flood_probability", 0) < fastest.get("mean_flood_probability", 0)):
        return {
            "selected_profile": "safest",
            "verdict": "SAFEST ROUTE RECOMMENDED",
            "rationale": (
                f"FASTEST route crosses {f_high:.1f} km of high flood exposure. "
                f"SAFEST route reduces high-risk exposure to {s_high:.1f} km with a {time_diff} min detour."
            ),
        }
    elif f_high == 0.0 and f_crit == 0.0:
        return {
            "selected_profile": "fastest",
            "verdict": "FASTEST ROUTE RECOMMENDED",
            "rationale": "The fastest path has minimal predicted flood hazard and provides the quickest evacuation time.",
        }
    else:
        return {
            "selected_profile": "balanced",
            "verdict": "BALANCED ROUTE RECOMMENDED",
            "rationale": "Balanced profile achieves acceptable hazard reduction while preserving reasonable travel time.",
        }
