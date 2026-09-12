from app.models.location import RiskLevel, RoadStatus, ShelterStatus, ZoneEvacuationStatus
from app.models.response import SimulationResult
from app.simulation.routes import find_safe_route
from app.world import CrisisWorld

EVAC_FRACTION = {
    RiskLevel.CRITICAL: 0.85,
    RiskLevel.HIGH: 0.55,
    RiskLevel.MODERATE: 0.15,
    RiskLevel.LOW: 0.0,
}


def _demand_for_zone(zone) -> int:
    if zone.evacuation_status == ZoneEvacuationStatus.COMPLETE:
        return 0
    base = EVAC_FRACTION.get(zone.risk_level, 0.0)
    if zone.evacuation_status in {ZoneEvacuationStatus.ORDERED, ZoneEvacuationStatus.IN_PROGRESS}:
        base = max(base, 0.7)
    if zone.evacuation_status == ZoneEvacuationStatus.ADVISED:
        base = max(base, 0.35)
    return int(round(zone.population * base))


def simulate_evacuation(world: CrisisWorld) -> SimulationResult:
    remaining_capacity = {
        shelter.id: shelter.available_capacity for shelter in world.shelters.values()
    }
    road_flow: dict[str, int] = {road.id: 0 for road in world.roads.values()}
    people_required = 0
    people_evacuated = 0
    bottlenecks: list[str] = []
    unsafe_routes: list[str] = []
    zone_results: list[dict] = []
    assignments: list[dict] = []
    max_time = 0.0

    priority_zones = sorted(
        world.zones.values(),
        key=lambda z: (z.risk_score, z.population),
        reverse=True,
    )

    for zone in priority_zones:
        demand = _demand_for_zone(zone)
        people_required += demand
        remaining = demand
        shelters = sorted(
            world.shelters.values(),
            key=lambda s: (
                0 if s.status != ShelterStatus.CLOSED else 1,
                -s.available_capacity,
            ),
        )
        for shelter in shelters:
            if remaining <= 0:
                break
            if shelter.status == ShelterStatus.CLOSED:
                continue
            free = remaining_capacity.get(shelter.id, 0)
            if free <= 0:
                continue
            route = find_safe_route(world, zone.id, shelter.zone_id)
            if not route.feasible:
                unsafe_routes.append(
                    f"No safe route {zone.id} -> shelter {shelter.id} ({shelter.name})"
                )
                continue
            capacity_on_path = min(
                (world.roads[rid].capacity - road_flow[rid] for rid in route.roads_used),
                default=free,
            )
            if route.roads_used and capacity_on_path <= 0:
                bottlenecks.append(
                    f"Route from {zone.name} to {shelter.name} is at road capacity"
                )
                continue
            movable = remaining if not route.roads_used else min(remaining, free, max(capacity_on_path, 0))
            movable = min(movable, free, remaining)
            if movable <= 0:
                continue
            remaining_capacity[shelter.id] -= movable
            remaining -= movable
            people_evacuated += movable
            max_time = max(max_time, route.estimated_travel_time_min)
            for rid in route.roads_used:
                road_flow[rid] += movable
                road = world.roads[rid]
                if road_flow[rid] > road.capacity:
                    bottlenecks.append(f"{road.name} exceeds flow capacity ({road_flow[rid]}/{road.capacity})")
                if road.status == RoadStatus.DANGEROUS:
                    unsafe_routes.append(f"{road.name} used despite dangerous status")
            assignments.append(
                {
                    "from_zone": zone.id,
                    "shelter_id": shelter.id,
                    "people": movable,
                    "route": route.roads_used,
                    "travel_time_min": route.estimated_travel_time_min,
                }
            )
        unmet = remaining
        zone_results.append(
            {
                "zone_id": zone.id,
                "zone_name": zone.name,
                "demand": demand,
                "evacuated": demand - unmet,
                "remaining": unmet,
            }
        )

    shelter_used = 0
    shelter_total = 0
    for shelter in world.shelters.values():
        used = shelter.available_capacity - remaining_capacity.get(shelter.id, 0)
        shelter.current_occupancy += used
        shelter_used += shelter.current_occupancy
        shelter_total += shelter.capacity

    remaining_people = max(0, people_required - people_evacuated)
    completion = (people_evacuated / people_required) if people_required else 1.0
    utilization = shelter_used / shelter_total if shelter_total else 0.0
    unmet_shelters = [
        sid for sid, cap in remaining_capacity.items() if cap <= 0
    ]

    return SimulationResult(
        disaster_type=world.crisis.disaster_type.value,
        people_requiring_evacuation=people_required,
        people_evacuated=people_evacuated,
        people_remaining=remaining_people,
        people_protected=people_evacuated,
        shelter_utilization=round(utilization, 4),
        evacuation_time_min=round(max_time, 2),
        evacuation_completion=round(completion, 4),
        bottlenecks=sorted(set(bottlenecks)),
        unsafe_routes=sorted(set(unsafe_routes)),
        unmet_demand={
            "people_not_evacuated": remaining_people,
            "full_shelters": unmet_shelters,
        },
        assignments=assignments,
        zone_results=zone_results,
        evidence=[
            f"{people_evacuated} of {people_required} people assigned to shelters",
            f"Peak modeled evacuation travel time {max_time:.1f} min",
        ],
    )
