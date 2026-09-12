from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.exceptions import InvalidScenarioError
from app.logging_config import logger
from app.models.location import RoadStatus, ShelterStatus
from app.models.response import Scenario, ScenarioChange
from app.models.resource import ResourceType
from app.services.crisis_service import CrisisService, crisis_service, parse_disaster
from app.simulation.engine import simulate_world
from app.simulation.resources import allocate_resources
from app.simulation.risk import apply_risk_to_world
from app.simulation.routes import find_safe_route
from app.world import CrisisWorld


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ScenarioService:
    def __init__(self, crisis_service: CrisisService):
        self.crisis_service = crisis_service
        self.history: list[Scenario] = []

    def list_scenarios(self, disaster: str | None = None) -> list[Scenario]:
        if not disaster:
            return list(self.history)
        dtype = parse_disaster(disaster)
        crisis_id = f"crisis_{dtype.value}"
        return [s for s in self.history if s.crisis_id == crisis_id]

    def apply_changes(self, world: CrisisWorld, changes: list[ScenarioChange]) -> CrisisWorld:
        clone = world.clone()
        for change in changes:
            self._apply_change(clone, change)
        apply_risk_to_world(clone)
        return clone

    def _apply_change(self, world: CrisisWorld, change: ScenarioChange) -> None:
        ctype = change.type
        target = change.target_id
        if ctype == "road_status_change":
            if not target or target not in world.roads:
                raise InvalidScenarioError(f"Unknown road '{target}'")
            if not change.new_status:
                raise InvalidScenarioError("road_status_change requires new_status")
            world.roads[target].status = RoadStatus(change.new_status)
            if world.roads[target].status == RoadStatus.BLOCKED:
                world.roads[target].risk_score = max(world.roads[target].risk_score, 90)
        elif ctype == "shelter_status_change":
            if not target or target not in world.shelters:
                raise InvalidScenarioError(f"Unknown shelter '{target}'")
            shelter = world.shelters[target]
            occupancy = shelter.current_occupancy
            if change.new_status == "full" or change.field == "full":
                occupancy = shelter.capacity
            if change.value is not None and change.field == "current_occupancy":
                occupancy = int(change.value)
            world.shelters[target] = shelter.model_copy(update={"current_occupancy": occupancy})
        elif ctype == "hospital_capacity_change":
            if not target or target not in world.hospitals:
                raise InvalidScenarioError(f"Unknown hospital '{target}'")
            hospital = world.hospitals[target]
            if change.delta is not None:
                hospital.available_beds = max(0, hospital.available_beds + int(change.delta))
            if change.value is not None:
                hospital.available_beds = max(0, min(hospital.total_beds, int(change.value)))
            hospital.available_beds = min(hospital.available_beds, hospital.total_beds)
        elif ctype == "rainfall_change":
            # Preserve the explicit scenario input for the flood ML predictor.
            current = world.crisis.disaster_parameters.get("scenario_rainfall_override_mm")
            if current is None:
                from app.services.weather_service import get_live_rainfall
                live = get_live_rainfall()
                current = float(live.get("rainfall_24h_mm") or world.crisis.disaster_parameters.get("rainfall_mm") or 0.0)
            if change.delta_percent is not None:
                next_value = float(current) * (1.0 + float(change.delta_percent) / 100.0)
            elif change.delta is not None:
                next_value = float(current) + float(change.delta)
            elif change.value is not None:
                next_value = float(change.value)
            else:
                raise InvalidScenarioError("rainfall_change requires delta_percent, delta or value")
            world.crisis.disaster_parameters["scenario_rainfall_override_mm"] = max(0.0, next_value)
            world.crisis.disaster_parameters["rainfall_mm"] = max(0.0, next_value)
        elif ctype == "wind_speed_change":
            self._scale_param(world, "wind_speed_kmh", change)
            if change.delta_percent:
                params = world.crisis.disaster_parameters
                params["intensity_index"] = min(
                    1.0,
                    float(params.get("intensity_index", 0.5)) * (1 + change.delta_percent / 200),
                )
        elif ctype == "water_level_change":
            self._scale_param(world, "water_level_m", change, zone_only=True)
            if change.delta is not None:
                world.crisis.disaster_parameters["river_water_level_m"] = float(
                    world.crisis.disaster_parameters.get("river_water_level_m", 0)
                ) + change.delta
        elif ctype == "magnitude_change":
            params = world.crisis.disaster_parameters
            if change.delta is not None:
                params["magnitude"] = float(params.get("magnitude", 0)) + change.delta
            if change.value is not None:
                params["magnitude"] = float(change.value)
            for zone in world.zones.values():
                current = float(zone.parameters.get("seismic_intensity", 0))
                zone.parameters["seismic_intensity"] = current + (change.delta or 0) * 0.6
        elif ctype == "resource_availability_change":
            self._change_resources(world, change)
        elif ctype == "population_change":
            if not target or target not in world.zones:
                raise InvalidScenarioError(f"Unknown zone '{target}'")
            zone = world.zones[target]
            if change.delta_percent is not None:
                zone.population = int(zone.population * (1 + change.delta_percent / 100))
            if change.delta is not None:
                zone.population = max(0, zone.population + int(change.delta))
            if change.value is not None:
                zone.population = max(0, int(change.value))
        elif ctype == "parameter_change" and target in world.zones and change.field:
            value = change.value
            if change.delta_percent is not None:
                current = float(world.zones[target].parameters.get(change.field, 0))
                value = current * (1 + change.delta_percent / 100)
            world.zones[target].parameters[change.field] = value
        else:
            raise InvalidScenarioError(f"Unsupported scenario change type '{ctype}'")

    def _scale_param(self, world: CrisisWorld, field: str, change: ScenarioChange, zone_only: bool = False) -> None:
        targets = [world.zones[change.target_id]] if change.target_id and change.target_id in world.zones else list(world.zones.values())
        for zone in targets:
            current = float(zone.parameters.get(field, 0))
            if change.delta_percent is not None:
                zone.parameters[field] = current * (1 + change.delta_percent / 100)
            elif change.delta is not None:
                zone.parameters[field] = max(0, current + change.delta)
            elif change.value is not None:
                zone.parameters[field] = float(change.value)
        if not zone_only and field == "rainfall_mm":
            params = world.crisis.disaster_parameters
            current = float(params.get("rainfall_mm", 0))
            if change.delta_percent is not None:
                params["rainfall_mm"] = current * (1 + change.delta_percent / 100)
            elif change.value is not None:
                params["rainfall_mm"] = float(change.value)
        if not zone_only and field == "wind_speed_kmh":
            params = world.crisis.disaster_parameters
            current = float(params.get("wind_speed_kmh", 0))
            if change.delta_percent is not None:
                params["wind_speed_kmh"] = current * (1 + change.delta_percent / 100)
            elif change.value is not None:
                params["wind_speed_kmh"] = float(change.value)

    def _change_resources(self, world: CrisisWorld, change: ScenarioChange) -> None:
        matches = []
        if change.target_id and change.target_id in world.resources:
            matches = [world.resources[change.target_id]]
        elif change.field:
            try:
                rtype = ResourceType(change.field)
            except ValueError as exc:
                raise InvalidScenarioError(f"Unknown resource type '{change.field}'") from exc
            matches = [r for r in world.resources.values() if r.type == rtype]
        if not matches:
            raise InvalidScenarioError("resource_availability_change needs target_id or resource type")
        remaining = int(change.delta or change.value or 0)
        # negative delta means units become unavailable
        if change.delta is not None and change.delta < 0:
            to_remove = abs(int(change.delta))
            for res in matches:
                take = min(to_remove, res.available_quantity)
                res.available_quantity -= take
                to_remove -= take
                if to_remove <= 0:
                    break
        elif change.value is not None:
            for res in matches:
                res.available_quantity = max(0, min(res.quantity, int(change.value)))
        else:
            for res in matches:
                res.available_quantity = max(0, min(res.quantity, res.available_quantity + remaining))

    def run_scenario(self, disaster: str, description: str, changes: list[ScenarioChange]) -> dict[str, Any]:
        world = self.crisis_service.get_world(disaster)
        before_world = world.clone()
        apply_risk_to_world(before_world)
        before_sim = simulate_world(before_world.clone())
        after = self.apply_changes(world, changes)
        after_sim = simulate_world(after.clone())
        before_risk = {z.id: z.risk_score for z in before_world.zones.values()}
        ml_flood_before = None
        ml_flood_after = None
        if disaster == "flood":
            before_probe = next(iter(before_world.zones.values()), None)
            after_probe = next(iter(after.zones.values()), None)
            if before_probe is not None:
                before_result = __import__("app.simulation.risk", fromlist=["calculate_zone_risk"]).calculate_zone_risk(before_world, before_probe)
                ml_flood_before = next((f for f in before_result.contributing_factors if f.get("name") == "ml_flood_severity"), None)
            if after_probe is not None:
                after_result = __import__("app.simulation.risk", fromlist=["calculate_zone_risk"]).calculate_zone_risk(after, after_probe)
                ml_flood_after = next((f for f in after_result.contributing_factors if f.get("name") == "ml_flood_severity"), None)
        changed_risk = []
        affected_zones = []
        for zone in after.zones.values():
            delta = zone.risk_score - before_risk.get(zone.id, zone.risk_score)
            if abs(delta) >= 0.5:
                changed_risk.append(
                    {
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "before": round(before_risk[zone.id], 2),
                        "after": round(zone.risk_score, 2),
                        "delta": round(delta, 2),
                    }
                )
                affected_zones.append(zone.id)

        changed_routes = []
        for origin in list(world.zones)[:4]:
            dest = list(world.shelters.values())[0].zone_id
            b = find_safe_route(before_world, origin, dest)
            a = find_safe_route(after, origin, dest)
            if b.roads_used != a.roads_used or b.feasible != a.feasible:
                changed_routes.append(
                    {
                        "origin": origin,
                        "destination": dest,
                        "before": b.model_dump(mode="json"),
                        "after": a.model_dump(mode="json"),
                    }
                )

        before_alloc = allocate_resources(before_world.clone())
        after_alloc = after_sim["allocation"]
        before_evac = before_sim["evacuation"]
        after_evac = after_sim["evacuation"]

        scenario = Scenario(
            id=f"scn_{uuid4().hex[:8]}",
            crisis_id=world.crisis.id,
            description=description,
            changes=changes,
            created_at=_now(),
            resulting_state={
                "average_risk": after_sim["average_risk"],
                "overall_score": after_sim["overall_score"],
            },
            simulation_result=after_evac.model_dump(mode="json"),
        )
        self.history.append(scenario)
        logger.info("Ran what-if scenario %s on %s", scenario.id, world.crisis.disaster_type.value)
        return {
            "scenario": scenario,
            "before_world": before_world,
            "after_world": after,
            "before_sim": before_sim,
            "after_sim": after_sim,
            "ml_flood": {"before": ml_flood_before, "after": ml_flood_after} if disaster == "flood" else None,
            "changed_risk": changed_risk,
            "affected_zones": affected_zones,
            "changed_routes": changed_routes,
            "changed_resource_allocation": {
                "before_score": before_alloc.allocation_score,
                "after_score": after_alloc.allocation_score,
                "before_unmet": before_alloc.unmet_demand,
                "after_unmet": after_alloc.unmet_demand,
            },
            "changed_evacuation": {
                "before": before_evac.model_dump(mode="json"),
                "after": after_evac.model_dump(mode="json"),
            },
        }

    def parse_natural_language(self, text: str, disaster: str) -> list[ScenarioChange]:
        world = self.crisis_service.get_world(disaster)
        raw = text.lower()
        changes: list[ScenarioChange] = []

        road_match = re.search(r"road[_\s-]*(\d+)", raw)
        if road_match and any(word in raw for word in ["block", "unavailable", "closed", "close"]):
            rid = f"road_{int(road_match.group(1)):02d}"
            if rid not in world.roads:
                rid = f"road_{road_match.group(1)}"
            changes.append(
                ScenarioChange(
                    type="road_status_change",
                    target_id=rid if rid in world.roads else f"road_{int(road_match.group(1)):02d}",
                    new_status="blocked",
                    description=f"Block {rid}",
                )
            )

        if "main shelter" in raw or "north hills stadium" in raw:
            if any(word in raw for word in ["full", "fills", "capacity"]):
                changes.append(
                    ScenarioChange(
                        type="shelter_status_change",
                        target_id="s02",
                        new_status="full",
                        description="Main shelter becomes full",
                    )
                )

        rain = re.search(r"rainfall.*?(increase[s]?|up|by)\s+(\d+)\s*%", raw) or re.search(
            r"rainfall increases by\s+(\d+)\s*%", raw
        )
        if "rainfall" in raw and re.search(r"(\d+)\s*%", raw):
            pct = float(re.search(r"(\d+)\s*%", raw).group(1))
            if any(w in raw for w in ["increase", "increases", "up"]):
                changes.append(ScenarioChange(type="rainfall_change", delta_percent=pct, description=f"Rainfall +{pct}%"))
            elif any(w in raw for w in ["decrease", "decreases", "down"]):
                changes.append(ScenarioChange(type="rainfall_change", delta_percent=-pct, description=f"Rainfall {pct}%"))

        if "wind" in raw and re.search(r"(\d+)\s*%", raw):
            pct = float(re.search(r"(\d+)\s*%", raw).group(1))
            sign = -1 if any(w in raw for w in ["decrease", "decreases"]) else 1
            changes.append(ScenarioChange(type="wind_speed_change", delta_percent=sign * pct, description=f"Wind {sign*pct}%"))

        mag = re.search(r"magnitude.*?(\d+(\.\d+)?)", raw)
        if "magnitude" in raw:
            if "higher" in raw or "increase" in raw:
                delta = float(mag.group(1)) if mag else 0.5
                if delta > 2:
                    delta = 0.5
                changes.append(ScenarioChange(type="magnitude_change", delta=delta, description=f"Magnitude +{delta}"))

        beds = re.search(r"(lose[s]?|lost|loses)\s+(\d+)\s+beds", raw) or re.search(r"(\d+)\s+beds", raw)
        if "hospital" in raw and beds:
            qty = int(beds.group(2) if beds.lastindex and beds.lastindex >= 2 else beds.group(1))
            target = "h01"
            if "coastal" in raw:
                target = "h02"
            changes.append(
                ScenarioChange(
                    type="hospital_capacity_change",
                    target_id=target,
                    delta=-qty,
                    description=f"Hospital {target} loses {qty} beds",
                )
            )

        amb = re.search(r"(\d+)\s+ambulance", raw)
        if amb and any(w in raw for w in ["unavailable", "offline", "lose", "lost"]):
            changes.append(
                ScenarioChange(
                    type="resource_availability_change",
                    field="ambulance",
                    delta=-int(amb.group(1)),
                    description=f"{amb.group(1)} ambulances unavailable",
                )
            )

        zone_match = re.search(r"zone\s*(\d+)", raw)
        if zone_match and "people" in raw and re.search(r"(\d+)\s*%", raw):
            zid = f"z{int(zone_match.group(1)):02d}"
            pct = float(re.search(r"(\d+)\s*%", raw).group(1))
            sign = 1 if any(w in raw for w in ["more", "increase"]) else -1
            changes.append(
                ScenarioChange(
                    type="population_change",
                    target_id=zid,
                    delta_percent=sign * pct,
                    description=f"{zid} population {sign*pct}%",
                )
            )

        if not changes:
            raise InvalidScenarioError(
                "Could not parse a structured scenario from the request. Try mentioning a road, shelter, rainfall, wind, magnitude, hospital, ambulance, or zone."
            )
        return changes


scenario_service = ScenarioService(crisis_service)
