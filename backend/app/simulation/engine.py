from typing import Any

from app.models.location import HospitalStatus, RoadStatus, ShelterStatus, ZoneEvacuationStatus
from app.models.resource import ResourceStatus
from app.models.response import PlanAction, ResponsePlan, SimulationResult
from app.simulation.evacuation import simulate_evacuation
from app.simulation.resources import allocate_resources
from app.simulation.risk import apply_risk_to_world
from app.simulation.routes import find_safe_route
from app.world import CrisisWorld


def _score_plan(evac: SimulationResult, alloc: SimulationResult, risk_reduction: float) -> float:
    completion = evac.evacuation_completion
    remaining_penalty = min(1.0, evac.people_remaining / max(evac.people_requiring_evacuation, 1))
    bottleneck_penalty = min(0.3, 0.04 * len(evac.bottlenecks))
    alloc_part = alloc.allocation_score / 100.0
    reduction_part = min(1.0, max(0.0, risk_reduction / 40.0))
    raw = (
        40 * completion
        + 20 * alloc_part
        + 25 * reduction_part
        + 15 * (1 - remaining_penalty)
        - 100 * bottleneck_penalty
    )
    return round(max(0, min(100, raw)), 2)


def apply_action(world: CrisisWorld, action: PlanAction) -> list[str]:
    notes: list[str] = []
    atype = action.type
    target = action.target_id
    qty = int(action.quantity or 1)

    if atype == "evacuate_zone" and target in world.zones:
        world.zones[target].evacuation_status = ZoneEvacuationStatus.ORDERED
        notes.append(f"Evacuation ordered for {target}")
    elif atype == "close_road" and target in world.roads:
        world.roads[target].status = RoadStatus.BLOCKED
        notes.append(f"Closed {target}")
    elif atype == "open_road" and target in world.roads:
        world.roads[target].status = RoadStatus.OPEN
        world.roads[target].risk_score = min(world.roads[target].risk_score, 35)
        notes.append(f"Opened {target}")
    elif atype == "open_shelter" and target in world.shelters:
        shelter = world.shelters[target]
        shelter.status = ShelterStatus.OPEN
        notes.append(f"Opened shelter {target}")
    elif atype == "dispatch_ambulance":
        notes.extend(_dispatch(world, "ambulance", target, qty))
    elif atype == "dispatch_rescue_team":
        notes.extend(_dispatch(world, "rescue_team", target, qty))
    elif atype == "deploy_water_pumps":
        notes.extend(_dispatch(world, "water_pump", target, qty))
        if target in world.zones:
            p = world.zones[target].parameters
            p["drainage_capacity_pct"] = min(100, float(p.get("drainage_capacity_pct", 40)) + 12 * qty)
            p["water_level_m"] = max(0, float(p.get("water_level_m", 0)) - 0.25 * qty)
    elif atype == "deploy_emergency_vehicles":
        notes.extend(_dispatch(world, "emergency_vehicle", target, qty))
    elif atype == "move_resources" and target in world.zones:
        notes.extend(_move_any_resource(world, target, qty))
    elif atype == "prioritize_hospitals" and target in world.hospitals:
        hospital = world.hospitals[target]
        hospital.status = HospitalStatus.OPERATIONAL
        hospital.available_beds = min(hospital.total_beds, hospital.available_beds + 10)
        notes.append(f"Prioritized hospital {target}")
    elif atype == "reroute_evacuation_traffic" and target in world.roads:
        if world.roads[target].status != RoadStatus.BLOCKED:
            world.roads[target].status = RoadStatus.OPEN
            world.roads[target].risk_score = max(0, world.roads[target].risk_score - 15)
        notes.append(f"Reroute priority on {target}")
    else:
        notes.append(f"No-op or unknown action {atype} on {target}")
    return notes


def _dispatch(world: CrisisWorld, resource_type: str, zone_id: str | None, qty: int) -> list[str]:
    notes = []
    remaining = qty
    for res in world.resources.values():
        if res.type.value != resource_type or res.available_quantity <= 0:
            continue
        take = min(remaining, res.available_quantity)
        res.available_quantity -= take
        remaining -= take
        if zone_id and zone_id in world.zones:
            res.zone_id = zone_id
            res.latitude = world.zones[zone_id].latitude
            res.longitude = world.zones[zone_id].longitude
            res.status = ResourceStatus.DEPLOYED
        notes.append(f"Dispatched {take} {resource_type} from {res.id} to {zone_id}")
        if remaining <= 0:
            break
    if remaining > 0:
        notes.append(f"Unmet {resource_type} dispatch: {remaining} units")
    return notes


def _move_any_resource(world: CrisisWorld, zone_id: str, qty: int) -> list[str]:
    moved = 0
    notes = []
    for res in world.resources.values():
        if res.available_quantity <= 0:
            continue
        take = min(qty - moved, res.available_quantity)
        res.zone_id = zone_id
        res.latitude = world.zones[zone_id].latitude
        res.longitude = world.zones[zone_id].longitude
        moved += take
        notes.append(f"Moved {res.id} toward {zone_id}")
        if moved >= qty:
            break
    return notes


def simulate_world(world: CrisisWorld) -> dict[str, Any]:
    before_avg = sum(z.risk_score for z in world.zones.values()) / max(len(world.zones), 1)
    risks = apply_risk_to_world(world)
    evac = simulate_evacuation(world)
    alloc = allocate_resources(world)
    after_avg = sum(z.risk_score for z in world.zones.values()) / max(len(world.zones), 1)
    risk_reduction = max(0.0, before_avg - after_avg)
    # Evacuation and pumps also count as protection even if zone scores don't drop much.
    protection_boost = min(18.0, evac.evacuation_completion * 12 + alloc.allocation_score * 0.05)
    total_reduction = risk_reduction + protection_boost
    overall = _score_plan(evac, alloc, total_reduction)
    evac.risk_reduction = round(total_reduction, 2)
    evac.resource_utilization = alloc.resource_utilization
    evac.allocation_score = alloc.allocation_score
    evac.response_time_min = alloc.response_time_min
    evac.overall_score = overall
    evac.assignments = evac.assignments + alloc.assignments
    evac.unmet_demand = {**evac.unmet_demand, **alloc.unmet_demand}
    return {
        "risks": [r.model_dump(mode="json") for r in risks],
        "evacuation": evac,
        "allocation": alloc,
        "overall_score": overall,
        "risk_reduction": round(total_reduction, 2),
        "average_risk": round(after_avg, 2),
    }


def simulate_plan(world: CrisisWorld, plan: ResponsePlan) -> dict[str, Any]:
    clone = world.clone()
    notes: list[str] = []
    for action in plan.actions:
        notes.extend(apply_action(clone, action))
    result = simulate_world(clone)
    evac: SimulationResult = result["evacuation"]
    plan.estimated_people_helped = evac.people_evacuated
    plan.estimated_time = max(evac.evacuation_time_min, evac.response_time_min)
    plan.estimated_risk_reduction = evac.risk_reduction
    plan.score = evac.overall_score
    plan.evidence = list(plan.evidence) + notes + evac.evidence
    return {
        "plan": plan,
        "simulation": evac,
        "notes": notes,
        "world_after": clone,
        "risks": result["risks"],
        "allocation": result["allocation"],
    }


def compare_plans(world: CrisisWorld, plans: list[ResponsePlan]) -> list[dict[str, Any]]:
    ranked = []
    for plan in plans:
        outcome = simulate_plan(world, plan)
        sim: SimulationResult = outcome["simulation"]
        ranked.append(
            {
                "plan_id": plan.id,
                "title": plan.title,
                "score": sim.overall_score,
                "risk_reduction": sim.risk_reduction,
                "people_protected": sim.people_evacuated,
                "evacuation_time_min": sim.evacuation_time_min,
                "resource_utilization": sim.resource_utilization,
                "weaknesses": sim.bottlenecks + sim.unsafe_routes,
                "bottlenecks": sim.bottlenecks,
                "simulation": sim.model_dump(mode="json"),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["ranking"] = index
    return ranked


def sample_corridor_routes(world: CrisisWorld, limit: int = 6) -> list[dict[str, Any]]:
    risky = [z for z in world.zones.values() if z.affected]
    safe = sorted(world.zones.values(), key=lambda z: z.risk_score)
    routes = []
    for zone in risky[:4]:
        for shelter in list(world.shelters.values())[:3]:
            result = find_safe_route(world, zone.id, shelter.zone_id)
            if result.feasible:
                routes.append(result.model_dump(mode="json"))
            if len(routes) >= limit:
                return routes
    if not routes and safe:
        fallback = find_safe_route(world, safe[-1].id, safe[0].id)
        routes.append(fallback.model_dump(mode="json"))
    return routes
