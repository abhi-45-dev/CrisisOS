from app.models.crisis import DisasterType
from app.models.location import RiskLevel, RoadStatus, Zone, ZoneEvacuationStatus
from app.models.response import RiskResult
from app.world import CrisisWorld


def score_to_level(score: float) -> RiskLevel:
    if score >= 80:
        return RiskLevel.CRITICAL
    if score >= 60:
        return RiskLevel.HIGH
    if score >= 35:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def _clamp(value: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, value))


def _common_factors(world: CrisisWorld, zone: Zone) -> tuple[float, list[dict], list[str]]:
    connected = [
        road
        for road in world.roads.values()
        if zone.id in (road.from_zone, road.to_zone)
    ]
    blocked = sum(1 for r in connected if r.status == RoadStatus.BLOCKED)
    dangerous = sum(1 for r in connected if r.status == RoadStatus.DANGEROUS)
    access = _clamp(blocked * 28 + dangerous * 16 + (20 if not connected else 0))
    exposure = _clamp(zone.population / 450.0)
    vuln = _clamp(zone.vulnerability * 100)
    nearby_resources = sum(
        1
        for res in world.resources.values()
        if res.zone_id == zone.id and res.available_quantity > 0
    )
    resource_gap = _clamp(40 - nearby_resources * 12)
    infra = _clamp(blocked * 22 + dangerous * 14)
    factors = [
        {"name": "population_exposure", "score": round(exposure, 2)},
        {"name": "vulnerability", "score": round(vuln, 2)},
        {"name": "infrastructure_condition", "score": round(infra, 2)},
        {"name": "accessibility", "score": round(access, 2)},
        {"name": "local_resource_gap", "score": round(resource_gap, 2)},
    ]
    evidence = [
        f"{zone.name} population {zone.population}",
        f"{blocked} blocked and {dangerous} dangerous roads adjacent to {zone.id}",
        f"{nearby_resources} resource caches currently in {zone.name}",
    ]
    return 0.12 * exposure + 0.1 * vuln + 0.08 * infra + 0.08 * access + 0.05 * resource_gap, factors, evidence


def _flood_score(world: CrisisWorld, zone: Zone) -> tuple[float, list[dict], list[str]]:
    p = zone.parameters
    rainfall = _clamp(float(p.get("rainfall_mm", 0)) / 2.2)
    water = _clamp(float(p.get("water_level_m", 0)) / 0.05)
    water_change = _clamp(float(p.get("water_level_change_m", 0)) / 0.012)
    drainage = _clamp(100 - float(p.get("drainage_capacity_pct", 70)))
    inundation = _clamp(float(p.get("inundation_risk", 0)) * 100)
    specific = 0.22 * water + 0.14 * rainfall + 0.12 * drainage + 0.14 * inundation + 0.08 * water_change
    factors = [
        {"name": "rainfall", "score": round(rainfall, 2), "value": p.get("rainfall_mm")},
        {"name": "water_level", "score": round(water, 2), "value": p.get("water_level_m")},
        {"name": "water_level_change", "score": round(water_change, 2), "value": p.get("water_level_change_m")},
        {"name": "drainage_capacity_deficit", "score": round(drainage, 2), "value": p.get("drainage_capacity_pct")},
        {"name": "inundation_risk", "score": round(inundation, 2), "value": p.get("inundation_risk")},
    ]
    evidence = [
        f"Rainfall {p.get('rainfall_mm', 0)} mm, water level {p.get('water_level_m', 0)} m",
        f"Drainage capacity {p.get('drainage_capacity_pct', 0)}%",
    ]
    return specific, factors, evidence


def _cyclone_score(world: CrisisWorld, zone: Zone) -> tuple[float, list[dict], list[str]]:
    p = zone.parameters
    params = world.crisis.disaster_parameters
    wind = _clamp(float(p.get("wind_speed_kmh", params.get("wind_speed_kmh", 0))) / 1.7)
    rain = _clamp(float(p.get("rainfall_mm", params.get("rainfall_mm", 0))) / 2.0)
    intensity = _clamp(float(params.get("intensity_index", 0.5)) * 100)
    distance = float(p.get("storm_distance_km", 20))
    proximity = _clamp(100 - distance * 2.2)
    infra = _clamp(float(p.get("infrastructure_risk", 0)) * 100)
    flood_c = _clamp(float(p.get("flooding_contribution", 0)) * 100)
    specific = 0.2 * wind + 0.12 * rain + 0.12 * intensity + 0.12 * proximity + 0.12 * infra + 0.08 * flood_c
    factors = [
        {"name": "wind_speed", "score": round(wind, 2), "value": p.get("wind_speed_kmh")},
        {"name": "rainfall", "score": round(rain, 2), "value": p.get("rainfall_mm")},
        {"name": "storm_intensity", "score": round(intensity, 2), "value": params.get("intensity_index")},
        {"name": "storm_proximity", "score": round(proximity, 2), "value": distance},
        {"name": "infrastructure_risk", "score": round(infra, 2), "value": p.get("infrastructure_risk")},
        {"name": "flooding_contribution", "score": round(flood_c, 2), "value": p.get("flooding_contribution")},
    ]
    evidence = [
        f"Local wind {p.get('wind_speed_kmh', 0)} km/h, storm distance {distance} km",
        f"Infrastructure risk {p.get('infrastructure_risk', 0)}",
    ]
    return specific, factors, evidence


def _earthquake_score(world: CrisisWorld, zone: Zone) -> tuple[float, list[dict], list[str]]:
    p = zone.parameters
    params = world.crisis.disaster_parameters
    magnitude = _clamp(float(params.get("magnitude", 0)) / 0.09)
    intensity = _clamp(float(p.get("seismic_intensity", 0)) / 0.08)
    distance = float(p.get("distance_from_epicenter_km", 10))
    proximity = _clamp(100 - distance * 8)
    bldg = _clamp(float(p.get("building_vulnerability", zone.vulnerability)) * 100)
    specific = 0.18 * magnitude + 0.22 * intensity + 0.16 * proximity + 0.16 * bldg
    factors = [
        {"name": "magnitude", "score": round(magnitude, 2), "value": params.get("magnitude")},
        {"name": "seismic_intensity", "score": round(intensity, 2), "value": p.get("seismic_intensity")},
        {"name": "distance_from_epicenter", "score": round(proximity, 2), "value": distance},
        {"name": "building_vulnerability", "score": round(bldg, 2), "value": p.get("building_vulnerability")},
    ]
    evidence = [
        f"Seismic intensity {p.get('seismic_intensity', 0)} at {distance} km from epicenter",
        f"Event magnitude {params.get('magnitude')}",
    ]
    return specific, factors, evidence


def calculate_zone_risk(world: CrisisWorld, zone: Zone) -> RiskResult:
    common, common_factors, common_evidence = _common_factors(world, zone)
    dtype = world.crisis.disaster_type
    if dtype == DisasterType.FLOOD:
        specific, factors, evidence = _flood_score(world, zone)
    elif dtype == DisasterType.CYCLONE:
        specific, factors, evidence = _cyclone_score(world, zone)
    else:
        specific, factors, evidence = _earthquake_score(world, zone)
    score = _clamp(common + specific)
    level = score_to_level(score)
    return RiskResult(
        zone_id=zone.id,
        zone_name=zone.name,
        risk_score=round(score, 2),
        risk_level=level,
        contributing_factors=factors + common_factors,
        evidence=evidence + common_evidence,
        disaster_type=dtype.value,
        population=zone.population,
        affected=level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or score >= 55,
    )


def apply_risk_to_world(world: CrisisWorld) -> list[RiskResult]:
    results: list[RiskResult] = []
    for zone in world.zones.values():
        result = calculate_zone_risk(world, zone)
        zone.risk_score = result.risk_score
        zone.risk_level = result.risk_level
        zone.affected = result.affected
        if result.risk_level == RiskLevel.CRITICAL and zone.evacuation_status == ZoneEvacuationStatus.NONE:
            zone.evacuation_status = ZoneEvacuationStatus.ORDERED
        elif result.risk_level == RiskLevel.HIGH and zone.evacuation_status == ZoneEvacuationStatus.NONE:
            zone.evacuation_status = ZoneEvacuationStatus.ADVISED
        results.append(result)
    results.sort(key=lambda item: item.risk_score, reverse=True)
    return results
