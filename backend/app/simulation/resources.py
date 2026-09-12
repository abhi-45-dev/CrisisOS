from app.geo import haversine_km
from app.models.location import RiskLevel
from app.models.resource import ResourceStatus, ResourceType
from app.models.response import SimulationResult
from app.simulation.routes import find_safe_route
from app.world import CrisisWorld

PRIORITY_TYPES = {
    "flood": [ResourceType.WATER_PUMP, ResourceType.RESCUE_TEAM, ResourceType.EMERGENCY_VEHICLE, ResourceType.FOOD_SUPPLY],
    "cyclone": [ResourceType.RESCUE_TEAM, ResourceType.EMERGENCY_VEHICLE, ResourceType.MEDICAL_TEAM, ResourceType.FOOD_SUPPLY],
    "earthquake": [ResourceType.RESCUE_TEAM, ResourceType.AMBULANCE, ResourceType.MEDICAL_TEAM, ResourceType.FIRE_TRUCK],
}


def _need_for_zone(zone, disaster: str) -> dict[ResourceType, int]:
    intensity = zone.risk_score / 100.0
    pop_factor = max(1, zone.population // 8000)
    base = max(1, int(round(pop_factor * intensity * 2)))
    needs: dict[ResourceType, int] = {}
    for rtype in PRIORITY_TYPES.get(disaster, list(ResourceType)):
        needs[rtype] = base
        if zone.risk_level == RiskLevel.CRITICAL:
            needs[rtype] += 1
    return needs


def allocate_resources(world: CrisisWorld) -> SimulationResult:
    disaster = world.crisis.disaster_type.value
    zones = sorted(world.zones.values(), key=lambda z: z.risk_score * (1 + z.population / 50000), reverse=True)
    assignments: list[dict] = []
    unmet: dict[str, dict[str, int]] = {}
    response_times: list[float] = []
    used_units = 0
    total_units = sum(res.quantity for res in world.resources.values())

    for zone in zones:
        if zone.risk_level == RiskLevel.LOW:
            continue
        needs = _need_for_zone(zone, disaster)
        for rtype, qty in needs.items():
            still = qty
            candidates = [
                res
                for res in world.resources.values()
                if res.type == rtype and res.available_quantity > 0 and res.status != ResourceStatus.UNAVAILABLE
            ]
            candidates.sort(
                key=lambda res: haversine_km(zone.latitude, zone.longitude, res.latitude, res.longitude)
            )
            for res in candidates:
                if still <= 0:
                    break
                take = min(still, res.available_quantity)
                res.available_quantity -= take
                still -= take
                used_units += take
                route = find_safe_route(world, res.zone_id, zone.id)
                eta = (
                    route.estimated_travel_time_min
                    if route.feasible
                    else haversine_km(zone.latitude, zone.longitude, res.latitude, res.longitude) * 4
                )
                response_times.append(eta)
                assignments.append(
                    {
                        "resource_id": res.id,
                        "resource_type": rtype.value,
                        "target_zone": zone.id,
                        "quantity": take,
                        "estimated_response_time_min": round(eta, 2),
                        "route_feasible": route.feasible,
                    }
                )
            if still > 0:
                unmet.setdefault(zone.id, {})[rtype.value] = still

    utilization = used_units / total_units if total_units else 0.0
    avg_time = sum(response_times) / len(response_times) if response_times else 0.0
    coverage = 1.0 - (len(unmet) / max(len([z for z in zones if z.risk_level != RiskLevel.LOW]), 1))
    score = round(100 * (0.55 * coverage + 0.25 * utilization + 0.2 * max(0, 1 - avg_time / 60)), 2)

    return SimulationResult(
        disaster_type=disaster,
        resource_utilization=round(utilization, 4),
        response_time_min=round(avg_time, 2),
        allocation_score=score,
        assignments=assignments,
        unmet_demand={"resource_shortages": unmet},
        evidence=[
            f"Assigned {used_units} resource units across {len(assignments)} deployments",
            f"Average modeled response time {avg_time:.1f} min",
        ],
    )
