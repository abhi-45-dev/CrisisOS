from __future__ import annotations

from typing import Any, Literal

from app.geo import haversine_km
from app.providers.boundary_provider import boundary_provider
from app.providers.infrastructure_provider import evacuation_facility_provider, hospital_provider
from app.routing.exposure import analyze_route_exposure
from app.routing.provider import RoutingProfile, osm_routing_provider
from app.services.hazard_service import hazard_service

# Tamil translations for emergency directives
TAMIL_STRINGS = {
    "NO_ROUTE_CONFIDENTLY_RECOMMENDED": "பாதுகாப்பான பாதை உறுதிப்படுத்தப்படவில்லை: தீவிர வெள்ள அபாயம் நிலவுகிறது. உயரமான இடத்தில் பாதுகாப்பாக இருக்கவும்.",
    "SAFEST_DESTINATION_RECOMMENDED": "பாதுகாப்பான தங்குமிடம் / மருத்துவமனை பரிந்துரைக்கப்படுகிறது.",
    "HIGH_FLOOD_EXPOSURE_WARNING": "எச்சரிக்கை: இந்த பாதையில் கடுமையான வெள்ள அபாயம் கணிக்கப்பட்டுள்ளது.",
    "PROCEED_WITH_CAUTION": "எச்சரிக்கையுடன் செல்லவும். உள்ளூர் பேரிடர் மேலாண்மை வழிகாட்டுதலைப் பின்பற்றவும்.",
}


class SurvivorService:
    """Survivor evacuation intelligence engine for Tamil Nadu."""

    def evaluate_evacuation(
        self,
        survivor_lat: float,
        survivor_lon: float,
        preferred_type: Literal["all", "hospital", "shelter"] = "all",
        language: Literal["en", "ta"] = "en",
        radius_km: float = 35.0,
    ) -> dict[str, Any]:
        # 1. Validate coordinate
        boundary_provider.validate_coordinates(survivor_lat, survivor_lon)

        # 2. Local hazard at survivor location
        survivor_hazard = hazard_service.predict_coordinate(survivor_lat, survivor_lon)

        # 3. Discover candidates
        candidates = []
        if preferred_type in ("all", "hospital"):
            hosps = hospital_provider.get_nearby(survivor_lat, survivor_lon, radius_km=radius_km, limit=5)
            for h in hosps:
                candidates.append({**h, "category": "hospital"})

        if preferred_type in ("all", "shelter"):
            shelters = evacuation_facility_provider.get_nearby(survivor_lat, survivor_lon, radius_km=radius_km, limit=5)
            for s in shelters:
                candidates.append({**s, "category": "shelter"})

        if not candidates:
            return {
                "survivor_location": [survivor_lat, survivor_lon],
                "survivor_hazard": survivor_hazard,
                "recommended_destination": None,
                "candidate_options": [],
                "verdict": "NO_FACILITY_WITHIN_RANGE",
                "rationale": f"No public hospital or evacuation facility found within {radius_km} km radius.",
                "disclaimer": "Decision support only. Not an authoritative emergency dispatch instruction.",
            }

        # 4. Evaluate routes and exposure for each candidate
        evaluated_candidates = []
        for cand in candidates:
            # Site hazard at destination
            site_hazard = hazard_service.predict_coordinate(cand["latitude"], cand["longitude"])
            if site_hazard["flood_probability"] >= 0.85:
                # Reject destination if the facility itself is submerged
                continue

            # Calculate fastest and safest routes
            fastest_raw = osm_routing_provider.calculate_route(
                survivor_lat, survivor_lon,
                cand["latitude"], cand["longitude"],
                profile=RoutingProfile.FASTEST,
            )
            safest_raw = osm_routing_provider.calculate_route(
                survivor_lat, survivor_lon,
                cand["latitude"], cand["longitude"],
                profile=RoutingProfile.SAFEST,
            )

            fastest_exp = analyze_route_exposure(fastest_raw)
            safest_exp = analyze_route_exposure(safest_raw)

            evaluated_candidates.append({
                "facility": cand,
                "destination_hazard": site_hazard,
                "fastest_route": fastest_exp,
                "safest_route": safest_exp,
                # Composite safety penalty score
                "penalty_score": (
                    cand["distance_km"] * 1.0 +
                    safest_exp.get("critical_risk_distance_km", 0.0) * 15.0 +
                    safest_exp.get("high_risk_distance_km", 0.0) * 5.0
                ),
            })

        if not evaluated_candidates:
            verdict = "NO ROUTE CONFIDENTLY RECOMMENDED"
            rationale = "All nearby facilities or their access corridors exceed acceptable flood risk thresholds."
            return {
                "survivor_location": [survivor_lat, survivor_lon],
                "survivor_hazard": survivor_hazard,
                "recommended_destination": None,
                "candidate_options": [],
                "verdict": verdict,
                "verdict_ta": TAMIL_STRINGS["NO_ROUTE_CONFIDENTLY_RECOMMENDED"] if language == "ta" else None,
                "rationale": rationale,
                "disclaimer": "Decision support only. Not an authoritative emergency dispatch instruction.",
            }

        # Sort by penalty score (best safety-to-distance tradeoff)
        evaluated_candidates.sort(key=lambda c: c["penalty_score"])
        best = evaluated_candidates[0]

        # Formulate human-understandable explanation
        rationale = self._explain_recommendation(best, evaluated_candidates)
        rationale_ta = None
        if language == "ta":
            rationale_ta = f"{best['facility']['name']} - {TAMIL_STRINGS['SAFEST_DESTINATION_RECOMMENDED']}"

        return {
            "survivor_location": [survivor_lat, survivor_lon],
            "survivor_hazard": survivor_hazard,
            "recommended_destination": best["facility"],
            "recommended_route": best["safest_route"],
            "fastest_alternative_route": best["fastest_route"],
            "candidate_options": evaluated_candidates[:3],
            "verdict": "RECOMMENDED",
            "rationale": rationale,
            "rationale_ta": rationale_ta,
            "disclaimer": "Decision support prototype. If in immediate peril, contact State Emergency Operations Centre (1070/112).",
        }

    def _explain_recommendation(self, best: dict, all_candidates: list[dict]) -> str:
        best_name = best["facility"]["name"]
        best_dist = best["facility"]["distance_km"]
        best_crit = best["safest_route"].get("critical_risk_distance_km", 0.0)

        # Check if there was a geographically closer option with higher flood risk
        closest = min(all_candidates, key=lambda c: c["facility"]["distance_km"])
        if closest["facility"]["id"] != best["facility"]["id"]:
            closer_name = closest["facility"]["name"]
            dist_diff = round(best_dist - closest["facility"]["distance_km"], 1)
            closer_risk = closest["fastest_route"].get("high_risk_distance_km", 0.0) + closest["fastest_route"].get("critical_risk_distance_km", 0.0)
            return (
                f"{closer_name} is {dist_diff} km closer, but its route traverses {closer_risk:.1f} km "
                f"of severe predicted flood hazard. {best_name} is therefore recommended because its "
                f"safest route has only {best_crit:.1f} km of critical exposure."
            )

        return (
            f"{best_name} is the optimal destination ({best_dist:.1f} km away). "
            f"Its route has minimal predicted road flood exposure ({best_crit:.1f} km critical risk) "
            "over the real road network."
        )


survivor_service = SurvivorService()
