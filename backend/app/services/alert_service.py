from datetime import datetime, timezone
from uuid import uuid4

from app.models.location import Alert, RiskLevel, RoadStatus, ShelterStatus
from app.models.resource import ResourceStatus, ResourceType
from app.world import CrisisWorld


def generate_alerts(world: CrisisWorld) -> list[Alert]:
    alerts: list[Alert] = []
    now = datetime.now(timezone.utc).isoformat()
    disaster = world.crisis.disaster_type.value
    params = world.crisis.disaster_parameters

    for zone in world.zones.values():
        if zone.risk_level == RiskLevel.CRITICAL:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.CRITICAL,
                    type="critical_zone_risk",
                    message=f"{zone.name} is at critical risk ({zone.risk_score:.1f}/100).",
                    timestamp=now,
                    affected_zone=zone.id,
                    evidence=[f"risk_score={zone.risk_score}", f"population={zone.population}"],
                    disaster_type=disaster,
                )
            )

    if disaster == "flood":
        if float(params.get("water_level_change_m", 0)) >= 0.4:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.HIGH,
                    type="rapidly_rising_water",
                    message=f"River water rising quickly ({params.get('water_level_change_m')} m change).",
                    timestamp=now,
                    evidence=[f"river_water_level_m={params.get('river_water_level_m')}"],
                    disaster_type=disaster,
                )
            )
        for zone in world.zones.values():
            if float(zone.parameters.get("water_level_change_m", 0)) >= 0.5:
                alerts.append(
                    Alert(
                        id=f"al_{uuid4().hex[:8]}",
                        severity=RiskLevel.HIGH,
                        type="rapidly_rising_water",
                        message=f"Water level in {zone.name} rose {zone.parameters.get('water_level_change_m')} m.",
                        timestamp=now,
                        affected_zone=zone.id,
                        evidence=[f"water_level_m={zone.parameters.get('water_level_m')}"],
                        disaster_type=disaster,
                    )
                )

    if disaster == "cyclone" and float(params.get("wind_speed_kmh", 0)) >= 140:
        alerts.append(
            Alert(
                id=f"al_{uuid4().hex[:8]}",
                severity=RiskLevel.CRITICAL,
                type="cyclone_intensification",
                message=f"Cyclone winds at {params.get('wind_speed_kmh')} km/h with intensity {params.get('intensity')}.",
                timestamp=now,
                evidence=[f"intensity_index={params.get('intensity_index')}"],
                disaster_type=disaster,
            )
        )

    if disaster == "earthquake":
        for zone in world.zones.values():
            if float(zone.parameters.get("seismic_intensity", 0)) >= 6.5:
                alerts.append(
                    Alert(
                        id=f"al_{uuid4().hex[:8]}",
                        severity=RiskLevel.CRITICAL,
                        type="earthquake_high_intensity_zone",
                        message=f"{zone.name} seismic intensity {zone.parameters.get('seismic_intensity')}.",
                        timestamp=now,
                        affected_zone=zone.id,
                        evidence=[f"magnitude={params.get('magnitude')}"],
                        disaster_type=disaster,
                    )
                )

    for road in world.roads.values():
        if road.status == RoadStatus.BLOCKED:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.HIGH,
                    type="blocked_evacuation_route",
                    message=f"{road.name} is blocked between {road.from_zone} and {road.to_zone}.",
                    timestamp=now,
                    evidence=[f"road_id={road.id}", f"status={road.status.value}"],
                    disaster_type=disaster,
                )
            )

    for shelter in world.shelters.values():
        util = shelter.current_occupancy / max(shelter.capacity, 1)
        if util >= 0.8 or shelter.status == ShelterStatus.FULL:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.HIGH if util >= 0.95 else RiskLevel.MODERATE,
                    type="shelter_near_capacity",
                    message=f"{shelter.name} is at {util:.0%} occupancy.",
                    timestamp=now,
                    affected_zone=shelter.zone_id,
                    evidence=[f"occupancy={shelter.current_occupancy}/{shelter.capacity}"],
                    disaster_type=disaster,
                )
            )

    for hospital in world.hospitals.values():
        remaining = hospital.available_beds / max(hospital.total_beds, 1)
        if remaining <= 0.2:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.HIGH,
                    type="hospital_near_capacity",
                    message=f"{hospital.name} has {hospital.available_beds} of {hospital.total_beds} beds free.",
                    timestamp=now,
                    affected_zone=hospital.zone_id,
                    evidence=[f"available_beds={hospital.available_beds}"],
                    disaster_type=disaster,
                )
            )

    ambulances = [r for r in world.resources.values() if r.type == ResourceType.AMBULANCE]
    available = sum(r.available_quantity for r in ambulances)
    if available <= 3:
        alerts.append(
            Alert(
                id=f"al_{uuid4().hex[:8]}",
                severity=RiskLevel.HIGH,
                type="resource_shortage",
                message=f"Only {available} ambulances available citywide.",
                timestamp=now,
                evidence=[f"status={[r.status.value for r in ambulances]}"],
                disaster_type=disaster,
            )
        )

    for resource in world.resources.values():
        if resource.status == ResourceStatus.UNAVAILABLE:
            alerts.append(
                Alert(
                    id=f"al_{uuid4().hex[:8]}",
                    severity=RiskLevel.MODERATE,
                    type="resource_shortage",
                    message=f"{resource.name} is unavailable.",
                    timestamp=now,
                    affected_zone=resource.zone_id,
                    evidence=[f"resource_id={resource.id}"],
                    disaster_type=disaster,
                )
            )

    severity_rank = {RiskLevel.CRITICAL: 0, RiskLevel.HIGH: 1, RiskLevel.MODERATE: 2, RiskLevel.LOW: 3}
    alerts.sort(key=lambda a: severity_rank[a.severity])
    return alerts
