from typing import Any

from app.exceptions import NotFoundError
from app.geo import haversine_km, items_within_radius, nearest
from app.models.crisis import DisasterType
from app.models.location import RiskLevel, RoadStatus
from app.services.alert_service import generate_alerts
from app.services.crisis_service import crisis_service
from app.services.scenario_service import scenario_service
from app.simulation.engine import sample_corridor_routes, simulate_plan, simulate_world
from app.simulation.evacuation import simulate_evacuation
from app.simulation.resources import allocate_resources
from app.simulation.risk import apply_risk_to_world, calculate_zone_risk
from app.simulation.routes import find_safe_route, find_safe_routes
from app.world import CrisisWorld


def get_crisis_state(disaster_type: str) -> dict[str, Any]:
    return crisis_service.get_crisis_state(disaster_type)


def get_zone_status(zone_id: str, disaster_type: str = "flood") -> dict[str, Any]:
    return crisis_service.get_zone_status(disaster_type, zone_id).model_dump(mode="json")


def get_road_status(road_id: str, disaster_type: str = "flood") -> dict[str, Any]:
    return crisis_service.get_road_status(disaster_type, road_id).model_dump(mode="json")


def get_hospital_capacity(hospital_id: str, disaster_type: str = "flood") -> dict[str, Any]:
    return crisis_service.get_hospital_capacity(disaster_type, hospital_id).model_dump(mode="json")


def get_shelter_capacity(shelter_id: str, disaster_type: str = "flood") -> dict[str, Any]:
    return crisis_service.get_shelter_capacity(disaster_type, shelter_id).model_dump(mode="json")


def get_available_resources(disaster_type: str = "flood") -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in crisis_service.get_available_resources(disaster_type)]


def find_safe_routes_tool(origin: str, destination: str, disaster_type: str = "flood") -> list[dict[str, Any]]:
    world = crisis_service.get_world(disaster_type)
    return [r.model_dump(mode="json") for r in find_safe_routes(world, origin, destination)]


def _coords(world: CrisisWorld, location: str | dict[str, Any]) -> tuple[float, float]:
    if isinstance(location, dict):
        return float(location["latitude"]), float(location["longitude"])
    if location in world.zones:
        zone = world.zones[location]
        return zone.latitude, zone.longitude
    raise NotFoundError(f"Unknown location '{location}'")


def find_nearest_shelter(location: str | dict[str, Any], disaster_type: str = "flood") -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    lat, lon = _coords(world, location)
    ranked = nearest(lat, lon, list(world.shelters.values()), limit=1)
    if not ranked:
        raise NotFoundError("No shelters available")
    shelter, dist = ranked[0]
    return {"shelter": shelter.model_dump(mode="json"), "distance_km": round(dist, 3)}


def find_nearest_hospital(location: str | dict[str, Any], disaster_type: str = "flood") -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    lat, lon = _coords(world, location)
    ranked = nearest(lat, lon, list(world.hospitals.values()), limit=1)
    hospital, dist = ranked[0]
    return {"hospital": hospital.model_dump(mode="json"), "distance_km": round(dist, 3)}


def find_nearest_resource(
    location: str | dict[str, Any],
    disaster_type: str = "flood",
    resource_type: str | None = None,
) -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    lat, lon = _coords(world, location)
    resources = list(world.resources.values())
    if resource_type:
        resources = [r for r in resources if r.type.value == resource_type]
    ranked = nearest(lat, lon, resources, limit=1)
    if not ranked:
        raise NotFoundError("No matching resource")
    resource, dist = ranked[0]
    return {"resource": resource.model_dump(mode="json"), "distance_km": round(dist, 3)}


def resources_within_radius(lat: float, lon: float, radius_km: float, disaster_type: str = "flood"):
    world = crisis_service.get_world(disaster_type)
    return [
        {"item": item.model_dump(mode="json"), "distance_km": round(dist, 3)}
        for item, dist in items_within_radius(lat, lon, list(world.resources.values()), radius_km)
    ]


def hospitals_within_radius(lat: float, lon: float, radius_km: float, disaster_type: str = "flood"):
    world = crisis_service.get_world(disaster_type)
    return [
        {"item": item.model_dump(mode="json"), "distance_km": round(dist, 3)}
        for item, dist in items_within_radius(lat, lon, list(world.hospitals.values()), radius_km)
    ]


def shelters_within_radius(lat: float, lon: float, radius_km: float, disaster_type: str = "flood"):
    world = crisis_service.get_world(disaster_type)
    return [
        {"item": item.model_dump(mode="json"), "distance_km": round(dist, 3)}
        for item, dist in items_within_radius(lat, lon, list(world.shelters.values()), radius_km)
    ]


def get_weather_data(disaster_type: str = "flood") -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    params = world.crisis.disaster_parameters
    return {
        "disaster_type": disaster_type,
        "rainfall_mm": params.get("rainfall_mm"),
        "wind_speed_kmh": params.get("wind_speed_kmh"),
        "wind_direction": params.get("wind_direction"),
        "water_level_m": params.get("river_water_level_m"),
        "magnitude": params.get("magnitude"),
        "parameters": params,
    }


def calculate_risk(disaster_type: str = "flood", zone_id: str | None = None) -> Any:
    world = crisis_service.get_world(disaster_type)
    if zone_id:
        if zone_id not in world.zones:
            raise NotFoundError(f"Unknown zone '{zone_id}'")
        result = calculate_zone_risk(world, world.zones[zone_id])
        world.zones[zone_id].risk_score = result.risk_score
        world.zones[zone_id].risk_level = result.risk_level
        return result.model_dump(mode="json")
    return [r.model_dump(mode="json") for r in apply_risk_to_world(world)]


def simulate_evacuation_tool(disaster_type: str = "flood") -> dict[str, Any]:
    world = crisis_service.clone_world(disaster_type)
    apply_risk_to_world(world)
    return simulate_evacuation(world).model_dump(mode="json")


def simulate_response(disaster_type: str, plan) -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    outcome = simulate_plan(world, plan)
    return {
        "plan": outcome["plan"].model_dump(mode="json"),
        "simulation": outcome["simulation"].model_dump(mode="json"),
        "notes": outcome["notes"],
    }


def run_what_if_scenario(disaster_type: str, description: str, changes) -> dict[str, Any]:
    return scenario_service.run_scenario(disaster_type, description, changes)


def affected_zones(disaster_type: str) -> list[dict[str, Any]]:
    world = crisis_service.get_world(disaster_type)
    return [z.model_dump(mode="json") for z in world.zones.values() if z.affected]


def safe_zones(disaster_type: str) -> list[dict[str, Any]]:
    world = crisis_service.get_world(disaster_type)
    return [
        z.model_dump(mode="json")
        for z in world.zones.values()
        if z.risk_level in {RiskLevel.LOW, RiskLevel.MODERATE} and not z.affected
    ]


def dangerous_roads(disaster_type: str) -> list[dict[str, Any]]:
    world = crisis_service.get_world(disaster_type)
    return [
        r.model_dump(mode="json")
        for r in world.roads.values()
        if r.status in {RoadStatus.BLOCKED, RoadStatus.DANGEROUS}
    ]


def map_state(disaster_type: str) -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    apply_risk_to_world(world)
    risks = [calculate_zone_risk(world, z).model_dump(mode="json") for z in world.zones.values()]
    risk_areas = [
        {
            "zone_id": z.id,
            "name": z.name,
            "latitude": z.latitude,
            "longitude": z.longitude,
            "risk_score": z.risk_score,
            "risk_level": z.risk_level.value,
            "affected": z.affected,
            "population": z.population,
            "parameters": z.parameters,
        }
        for z in world.zones.values()
        if z.risk_score >= 35
    ]
    epicenter = None
    if world.crisis.disaster_type == DisasterType.EARTHQUAKE:
        epicenter = {
            "latitude": world.crisis.disaster_parameters.get("epicenter_latitude"),
            "longitude": world.crisis.disaster_parameters.get("epicenter_longitude"),
            "magnitude": world.crisis.disaster_parameters.get("magnitude"),
        }
    storm = None
    if world.crisis.disaster_type == DisasterType.CYCLONE:
        storm = {
            "latitude": world.crisis.disaster_parameters.get("eye_latitude"),
            "longitude": world.crisis.disaster_parameters.get("eye_longitude"),
            "radius_km": world.crisis.disaster_parameters.get("storm_radius_km"),
            "wind_speed_kmh": world.crisis.disaster_parameters.get("wind_speed_kmh"),
        }
    return {
        "crisis": world.crisis.model_dump(mode="json"),
        "zones": [z.model_dump(mode="json") for z in world.zones.values()],
        "incidents": [i.model_dump(mode="json") for i in world.incidents],
        "roads": [r.model_dump(mode="json") for r in world.roads.values()],
        "hospitals": [h.model_dump(mode="json") for h in world.hospitals.values()],
        "shelters": [s.model_dump(mode="json") for s in world.shelters.values()],
        "resources": [r.model_dump(mode="json") for r in world.resources.values()],
        "risk_areas": risk_areas,
        "risks": risks,
        "evacuation_routes": sample_corridor_routes(world),
        "alerts": [a.model_dump(mode="json") for a in generate_alerts(world)],
        "disaster_overlay": {
            "epicenter": epicenter,
            "storm": storm,
            "flood_extent_zones": [z.id for z in world.zones.values() if z.affected]
            if world.crisis.disaster_type == DisasterType.FLOOD
            else [],
        },
    }


def dashboard_summary(disaster_type: str) -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    apply_risk_to_world(world)
    evac = simulate_evacuation(world.clone())
    alerts = generate_alerts(world)
    affected_pop = sum(z.population for z in world.zones.values() if z.affected)
    critical = [z.id for z in world.zones.values() if z.risk_level == RiskLevel.CRITICAL]
    high = [z.id for z in world.zones.values() if z.risk_level == RiskLevel.HIGH]
    blocked = [r.id for r in world.roads.values() if r.status == RoadStatus.BLOCKED]
    shelter_cap = sum(s.capacity for s in world.shelters.values())
    shelter_occ = sum(s.current_occupancy for s in world.shelters.values())
    beds = sum(h.available_beds for h in world.hospitals.values())
    resources = sum(r.available_quantity for r in world.resources.values())
    params = world.crisis.disaster_parameters
    specific: dict[str, Any] = {}
    if world.crisis.disaster_type == DisasterType.FLOOD:
        specific = {
            "water_level_m": params.get("river_water_level_m"),
            "rainfall_mm": params.get("rainfall_mm"),
            "affected_population": affected_pop,
            "flooded_zones": [z.id for z in world.zones.values() if z.affected],
            "blocked_roads": blocked,
            "evacuation_progress": evac.evacuation_completion,
            "shelter_occupancy": shelter_occ,
        }
    elif world.crisis.disaster_type == DisasterType.CYCLONE:
        specific = {
            "wind_speed_kmh": params.get("wind_speed_kmh"),
            "wind_direction": params.get("wind_direction"),
            "rainfall_mm": params.get("rainfall_mm"),
            "cyclone_intensity": params.get("intensity"),
            "affected_population": affected_pop,
            "infrastructure_risk": round(
                sum(float(z.parameters.get("infrastructure_risk", 0)) for z in world.zones.values())
                / max(len(world.zones), 1),
                3,
            ),
            "evacuation_progress": evac.evacuation_completion,
        }
    else:
        hospital_demand = max(0, affected_pop // 80)
        specific = {
            "magnitude": params.get("magnitude"),
            "epicenter": {
                "latitude": params.get("epicenter_latitude"),
                "longitude": params.get("epicenter_longitude"),
            },
            "seismic_intensity": params.get("peak_seismic_intensity"),
            "affected_population": affected_pop,
            "damaged_roads": [
                r.id
                for r in world.roads.values()
                if r.status in {RoadStatus.BLOCKED, RoadStatus.DANGEROUS}
            ],
            "hospital_demand": hospital_demand,
            "emergency_resource_demand": max(1, affected_pop // 5000),
        }
    return {
        "disaster_type": disaster_type,
        "overall_severity": world.crisis.severity,
        "status": world.crisis.status.value,
        "affected_population": affected_pop,
        "critical_zones": critical,
        "high_risk_zones": high,
        "active_incidents": len([i for i in world.incidents if i.status.value in {"active", "reported"}]),
        "blocked_roads": blocked,
        "available_shelters": [s.id for s in world.shelters.values() if s.available_capacity > 0],
        "shelter_utilization": round(shelter_occ / max(shelter_cap, 1), 4),
        "available_hospital_beds": beds,
        "available_emergency_resources": resources,
        "evacuation_progress": evac.evacuation_completion,
        "people_requiring_evacuation": evac.people_requiring_evacuation,
        "people_evacuated_modeled": evac.people_evacuated,
        "current_alerts": [a.model_dump(mode="json") for a in alerts],
        "metrics": specific,
    }


def briefing_context(disaster_type: str) -> dict[str, Any]:
    world = crisis_service.get_world(disaster_type)
    apply_risk_to_world(world)
    risks = [calculate_zone_risk(world, z) for z in world.zones.values()]
    risks.sort(key=lambda r: r.risk_score, reverse=True)
    return {
        "crisis": world.crisis.model_dump(mode="json"),
        "top_risks": [r.model_dump(mode="json") for r in risks[:6]],
        "blocked_roads": dangerous_roads(disaster_type),
        "resources": get_available_resources(disaster_type),
        "incidents": [i.model_dump(mode="json") for i in world.incidents],
        "dashboard": dashboard_summary(disaster_type),
        "weather": get_weather_data(disaster_type),
        "nearest_safe_shelter_from_highest_risk": find_nearest_shelter(risks[0].zone_id, disaster_type)
        if risks
        else None,
    }


__all__ = [
    "affected_zones",
    "briefing_context",
    "calculate_risk",
    "dangerous_roads",
    "dashboard_summary",
    "find_nearest_hospital",
    "find_nearest_resource",
    "find_nearest_shelter",
    "find_safe_routes_tool",
    "get_available_resources",
    "get_crisis_state",
    "get_hospital_capacity",
    "get_road_status",
    "get_shelter_capacity",
    "get_weather_data",
    "get_zone_status",
    "haversine_km",
    "hospitals_within_radius",
    "map_state",
    "resources_within_radius",
    "run_what_if_scenario",
    "safe_zones",
    "shelters_within_radius",
    "simulate_evacuation_tool",
    "simulate_response",
]
