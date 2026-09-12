from __future__ import annotations

from typing import Any
from app.services.hazard_service import hazard_service
from app.routing.provider import osm_routing_provider, RoutingProfile
from app.routing.exposure import analyze_route_exposure


class ScenarioService:
    """Model-based 'what-if' simulation engine for FloodNow TN."""

    def simulate_rainfall_delta(
        self,
        rainfall_delta_mm: float,
        evaluate_route_origin: list[float] | None = None,
        evaluate_route_destination: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Run genuine ML model inference comparing current baseline vs modified rainfall.
        Evaluates:
        1. Baseline flood hazard surface (unmodified weather/soil inputs).
        2. Scenario flood hazard surface (rainfall_delta_mm added into feature vector).
        3. Statistical summary of risk band escalations and probability deltas.
        4. Optional route exposure re-analysis under scenario hazard.
        """
        # 1. Baseline grid
        baseline_grid = hazard_service.get_statewide_hazard_grid(rainfall_override_mm=None, force_refresh=False)
        baseline_features = {f["id"]: f for f in baseline_grid["features"]}

        # 2. Scenario grid (reruns ML inference with rainfall delta)
        scenario_grid = hazard_service.get_statewide_hazard_grid(
            rainfall_override_mm=rainfall_delta_mm,
            force_refresh=True,
        )

        comparison_features = []
        escalated_cells_count = 0
        total_delta = 0.0
        max_delta = 0.0
        critically_exposed_cells = 0

        for sf in scenario_grid["features"]:
            cid = sf["id"]
            bf = baseline_features.get(cid)
            b_prob = bf["properties"]["flood_probability"] if bf else 0.0
            b_band = bf["properties"]["risk_band"] if bf else "LOW"

            s_prob = sf["properties"]["flood_probability"]
            s_band = sf["properties"]["risk_band"]

            prob_delta = round(s_prob - b_prob, 4)
            total_delta += prob_delta
            if prob_delta > max_delta:
                max_delta = prob_delta

            is_escalated = (s_band in ("HIGH", "CRITICAL") and b_band not in ("HIGH", "CRITICAL"))
            if is_escalated:
                escalated_cells_count += 1
            if s_band == "CRITICAL":
                critically_exposed_cells += 1

            merged_props = dict(sf["properties"])
            merged_props.update({
                "baseline_flood_probability": b_prob,
                "scenario_flood_probability": s_prob,
                "delta_flood_probability": prob_delta,
                "baseline_risk_band": b_band,
                "scenario_risk_band": s_band,
                "risk_band_escalated": is_escalated,
            })

            comparison_features.append({
                "type": "Feature",
                "id": cid,
                "properties": merged_props,
                "geometry": sf["geometry"],
            })

        n_cells = len(comparison_features) or 1
        avg_delta = round(total_delta / n_cells, 4)

        route_impact = None
        if evaluate_route_origin and evaluate_route_destination:
            # Re-evaluate routes under scenario hazard
            try:
                base_route = osm_routing_provider.calculate_route(
                    evaluate_route_origin[0], evaluate_route_origin[1],
                    evaluate_route_destination[0], evaluate_route_destination[1],
                    profile=RoutingProfile.SAFEST,
                    hazard_override_cells=baseline_grid,
                )
                scen_route = osm_routing_provider.calculate_route(
                    evaluate_route_origin[0], evaluate_route_origin[1],
                    evaluate_route_destination[0], evaluate_route_destination[1],
                    profile=RoutingProfile.SAFEST,
                    hazard_override_cells=scenario_grid,
                )
                route_impact = {
                    "baseline_exposure": analyze_route_exposure(base_route, hazard_grid=baseline_grid),
                    "scenario_exposure": analyze_route_exposure(scen_route, hazard_grid=scenario_grid),
                }
            except Exception as e:
                route_impact = {"error": str(e)}

        summary_narrative = (
            f"An increase of +{rainfall_delta_mm:.1f}mm rainfall across Tamil Nadu results in "
            f"{escalated_cells_count} grid zones escalating into HIGH or CRITICAL flood hazard. "
            f"Average statewide flood probability increased by +{avg_delta * 100:.1f}%, "
            f"with a maximum localized increase of +{max_delta * 100:.1f}%."
        )

        return {
            "scenario": {
                "type": "rainfall_delta",
                "rainfall_delta_mm": rainfall_delta_mm,
                "model_version": "FloodNow TN v1.0.0",
                "summary": summary_narrative,
                "metrics": {
                    "total_cells_evaluated": n_cells,
                    "escalated_cells_count": escalated_cells_count,
                    "critically_exposed_cells": critically_exposed_cells,
                    "average_probability_delta": avg_delta,
                    "maximum_probability_delta": round(max_delta, 4),
                },
            },
            "route_impact": route_impact,
            "geojson": {
                "type": "FeatureCollection",
                "metadata": {
                    "scenario_delta_mm": rainfall_delta_mm,
                    "cell_count": len(comparison_features),
                },
                "features": comparison_features,
            },
        }


scenario_service = ScenarioService()
