from __future__ import annotations

from typing import Any
from shapely.geometry import LineString, Point, shape

from app.geo import haversine_km


def analyze_route_exposure(
    route: dict[str, Any],
    hazard_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently evaluate route geometry against flood hazard surface.

    Returns:
    - distance_km
    - baseline_network_duration_min
    - max_flood_probability
    - mean_flood_probability
    - high_risk_distance_km (hazard >= 0.60)
    - critical_risk_distance_km (hazard >= 0.80)
    - hazard_segments
    - geometry
    """
    if not route.get("feasible"):
        return {
            **route,
            "max_flood_probability": 0.0,
            "mean_flood_probability": 0.0,
            "high_risk_distance_km": 0.0,
            "critical_risk_distance_km": 0.0,
            "hazard_segments": [],
            "risk_verdict": "NO_ROUTE",
        }

    coords = route.get("coordinates", [])
    if len(coords) < 2:
        return {
            **route,
            "max_flood_probability": 0.0,
            "mean_flood_probability": 0.0,
            "high_risk_distance_km": 0.0,
            "critical_risk_distance_km": 0.0,
            "hazard_segments": [],
            "risk_verdict": "CLEAR",
        }

    # Prepared spatial polygons for fast point-in-cell lookups
    prepared_cells = []
    if hazard_cells:
        for cell in hazard_cells:
            geom = cell.get("geometry")
            if geom:
                prepared_cells.append({
                    "poly": shape(geom),
                    "prob": float(cell.get("flood_probability") or cell.get("probability") or 0.0),
                    "risk_band": cell.get("risk_band", "low"),
                })

    def get_point_hazard(lon: float, lat: float) -> float:
        pt = Point(lon, lat)
        for cell in prepared_cells:
            if cell["poly"].contains(pt):
                return cell["prob"]
        return 0.0

    # Sample subsegments between consecutive coordinates
    segments = []
    sample_probs = []
    high_risk_km = 0.0
    critical_risk_km = 0.0

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        seg_dist = haversine_km(p1[1], p1[0], p2[1], p2[0])
        seg_line = LineString([p1, p2])

        seg_max_prob = 0.0
        seg_exposed_km = 0.0

        for cell in prepared_cells:
            if seg_line.intersects(cell["poly"]):
                prob = cell["prob"]
                seg_max_prob = max(seg_max_prob, prob)
                
                # Compute fractional distance inside polygon
                inter = seg_line.intersection(cell["poly"])
                if seg_line.length > 0:
                    fraction = min(1.0, max(0.0, inter.length / seg_line.length))
                else:
                    fraction = 1.0
                seg_exposed_km = max(seg_exposed_km, seg_dist * fraction)

        sample_probs.append(seg_max_prob)

        band = "low"
        if seg_max_prob >= 0.80:
            band = "critical"
            critical_risk_km += seg_exposed_km
            high_risk_km += seg_exposed_km
        elif seg_max_prob >= 0.60:
            band = "high"
            high_risk_km += seg_exposed_km
        elif seg_max_prob >= 0.35:
            band = "moderate"

        segments.append({
            "segment_index": i,
            "coordinates": [p1, p2],
            "distance_km": round(seg_dist, 3),
            "flood_probability": round(seg_max_prob, 4),
            "risk_band": band,
        })

    max_prob = max(sample_probs) if sample_probs else 0.0
    mean_prob = sum(sample_probs) / max(len(sample_probs), 1)

    # Risk verdict
    if critical_risk_km > 0.5:
        verdict = "CRITICAL_HAZARD"
    elif high_risk_km > 1.0 or max_prob >= 0.60:
        verdict = "ELEVATED_EXPOSURE"
    elif max_prob >= 0.35:
        verdict = "MODERATE_EXPOSURE"
    else:
        verdict = "SAFE"

    return {
        "profile": route["profile"],
        "feasible": route["feasible"],
        "total_distance_km": route["total_distance_km"],
        "baseline_network_duration_min": route["estimated_duration_min"],
        "max_flood_probability": round(max_prob, 4),
        "mean_flood_probability": round(mean_prob, 4),
        "high_risk_distance_km": round(high_risk_km, 2),
        "critical_risk_distance_km": round(critical_risk_km, 2),
        "risk_verdict": verdict,
        "road_names": route.get("road_names", []),
        "hazard_segments": segments,
        "geometry": route.get("geometry", {
            "type": "LineString",
            "coordinates": coords,
        }),
        "warnings": route.get("warnings", []),
    }
