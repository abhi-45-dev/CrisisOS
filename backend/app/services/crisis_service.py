from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings
from app.exceptions import InvalidDisasterError, InvalidEventError, NotFoundError
from app.logging_config import logger
from app.models.crisis import Crisis, DisasterType
from app.models.incident import Incident, IncidentStatus
from app.models.location import Hospital, HospitalStatus, Road, RoadStatus, Shelter, ShelterStatus, Zone
from app.models.resource import Resource, ResourceStatus
from app.simulation.risk import apply_risk_to_world
from app.world import CrisisWorld

DISASTER_TYPES = {item.value: item for item in DisasterType}


def parse_disaster(value: str) -> DisasterType:
    key = value.lower().strip()
    if key not in DISASTER_TYPES:
        raise InvalidDisasterError(
            f"Unsupported disaster type '{value}'. Use flood, cyclone, or earthquake.",
            {"supported": list(DISASTER_TYPES)},
        )
    return DISASTER_TYPES[key]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class CrisisService:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self._base_zones = _load_json(self.data_dir / "zones.json")
        self._base_roads = _load_json(self.data_dir / "roads.json")
        self._base_hospitals = _load_json(self.data_dir / "hospitals.json")
        self._base_shelters = _load_json(self.data_dir / "shelters.json")
        self._base_resources = _load_json(self.data_dir / "resources.json")
        self._incidents = _load_json(self.data_dir / "incidents.json")
        self._overlays = _load_json(self.data_dir / "disaster_overlays.json")
        self.worlds: dict[DisasterType, CrisisWorld] = {}
        self.reset()

    def reset(self) -> None:
        self.worlds = {
            DisasterType.FLOOD: self._build_world(DisasterType.FLOOD),
            DisasterType.CYCLONE: self._build_world(DisasterType.CYCLONE),
            DisasterType.EARTHQUAKE: self._build_world(DisasterType.EARTHQUAKE),
        }
        for world in self.worlds.values():
            apply_risk_to_world(world)
        logger.info("Loaded crisis worlds for flood, cyclone, and earthquake")

    def _build_world(self, disaster: DisasterType) -> CrisisWorld:
        overlay = deepcopy(self._overlays[disaster.value])
        crisis = Crisis.model_validate({k: overlay[k] for k in [
            "id", "name", "disaster_type", "status", "start_time",
            "current_time", "severity", "description", "disaster_parameters",
        ]})
        zones = {
            item["id"]: Zone.model_validate({**item, "parameters": {}})
            for item in deepcopy(self._base_zones)
        }
        for zone_id, params in overlay.get("zone_overrides", {}).items():
            if zone_id in zones:
                zones[zone_id].parameters = params
        roads = {item["id"]: Road.model_validate(item) for item in deepcopy(self._base_roads)}
        for road_id, values in overlay.get("road_overrides", {}).items():
            if road_id in roads:
                roads[road_id] = Road.model_validate({**roads[road_id].model_dump(), **values})
        hospitals = {item["id"]: Hospital.model_validate(item) for item in deepcopy(self._base_hospitals)}
        for hid, values in overlay.get("hospital_overrides", {}).items():
            if hid in hospitals:
                hospitals[hid] = Hospital.model_validate({**hospitals[hid].model_dump(), **values})
        shelters = {item["id"]: Shelter.model_validate(item) for item in deepcopy(self._base_shelters)}
        for sid, values in overlay.get("shelter_overrides", {}).items():
            if sid in shelters:
                shelters[sid] = Shelter.model_validate({**shelters[sid].model_dump(), **values})
        resources = {item["id"]: Resource.model_validate(item) for item in deepcopy(self._base_resources)}
        for rid, values in overlay.get("resource_overrides", {}).items():
            if rid in resources:
                resources[rid] = Resource.model_validate({**resources[rid].model_dump(), **values})
        incidents = [
            Incident.model_validate(item) for item in deepcopy(self._incidents.get(disaster.value, []))
        ]
        return CrisisWorld(
            crisis=crisis,
            zones=zones,
            roads=roads,
            hospitals=hospitals,
            shelters=shelters,
            resources=resources,
            incidents=incidents,
        )

    def get_world(self, disaster: str | DisasterType) -> CrisisWorld:
        dtype = disaster if isinstance(disaster, DisasterType) else parse_disaster(disaster)
        return self.worlds[dtype]

    def clone_world(self, disaster: str | DisasterType) -> CrisisWorld:
        return self.get_world(disaster).clone()

    def list_crises(self) -> list[Crisis]:
        return [world.crisis for world in self.worlds.values()]

    def get_crisis_state(self, disaster: str | DisasterType | None = None) -> dict[str, Any]:
        if disaster is None:
            return {
                "crises": [c.model_dump(mode="json") for c in self.list_crises()],
                "active": [dtype.value for dtype in self.worlds],
            }
        world = self.get_world(disaster)
        return world.snapshot()

    def get_zone_status(self, disaster: str, zone_id: str) -> Zone:
        world = self.get_world(disaster)
        if zone_id not in world.zones:
            raise NotFoundError(f"Unknown zone '{zone_id}'", {"id": zone_id})
        return world.zones[zone_id]

    def get_road_status(self, disaster: str, road_id: str) -> Road:
        world = self.get_world(disaster)
        if road_id not in world.roads:
            raise NotFoundError(f"Unknown road '{road_id}'", {"id": road_id})
        return world.roads[road_id]

    def get_hospital_capacity(self, disaster: str, hospital_id: str) -> Hospital:
        world = self.get_world(disaster)
        if hospital_id not in world.hospitals:
            raise NotFoundError(f"Unknown hospital '{hospital_id}'", {"id": hospital_id})
        return world.hospitals[hospital_id]

    def get_shelter_capacity(self, disaster: str, shelter_id: str) -> Shelter:
        world = self.get_world(disaster)
        if shelter_id not in world.shelters:
            raise NotFoundError(f"Unknown shelter '{shelter_id}'", {"id": shelter_id})
        return world.shelters[shelter_id]

    def get_available_resources(self, disaster: str) -> list[Resource]:
        world = self.get_world(disaster)
        return [r for r in world.resources.values() if r.available_quantity > 0]

    def get_incidents(self, disaster: str) -> list[Incident]:
        return list(self.get_world(disaster).incidents)

    def apply_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = (payload.get("type") or "").strip()
        disaster = payload.get("disaster_type") or payload.get("crisis_id", "")
        if str(disaster).startswith("crisis_"):
            disaster = str(disaster).replace("crisis_", "")
        world = self.get_world(str(disaster) if disaster else payload.get("disaster", "flood"))
        target_id = payload.get("target_id")
        logger.info("Applying event %s to %s", event_type, world.crisis.disaster_type.value)

        handlers = {
            "sensor_reading": self._event_sensor,
            "new_incident": self._event_incident,
            "road_closure": lambda w, p: self._set_road(w, p, RoadStatus.BLOCKED),
            "road_reopening": lambda w, p: self._set_road(w, p, RoadStatus.OPEN),
            "hospital_capacity_update": self._event_hospital,
            "shelter_capacity_update": self._event_shelter,
            "resource_status_update": self._event_resource,
            "water_level_change": self._event_water,
            "rainfall_change": self._event_rain,
            "cyclone_wind_change": self._event_wind,
            "earthquake_intensity_update": self._event_quake,
        }
        if event_type not in handlers:
            raise InvalidEventError(
                f"Unsupported event type '{event_type}'",
                {"supported": sorted(handlers)},
            )
        result = handlers[event_type](world, payload)
        apply_risk_to_world(world)
        return {
            "accepted": True,
            "event_type": event_type,
            "disaster_type": world.crisis.disaster_type.value,
            "target_id": target_id,
            "result": result,
        }

    def _event_sensor(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        zone_id = payload.get("target_id") or payload.get("zone_id")
        if zone_id not in world.zones:
            raise InvalidEventError("sensor reading requires a valid zone_id")
        readings = payload.get("readings") or payload.get("data") or {}
        world.zones[zone_id].parameters.update(readings)
        return {"zone_id": zone_id, "parameters": world.zones[zone_id].parameters}

    def _event_incident(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("incident") or payload
        incident = Incident(
            id=data.get("id") or f"inc_{uuid4().hex[:8]}",
            crisis_id=world.crisis.id,
            type=data.get("incident_type") or data.get("type") or "incident",
            description=data.get("description") or "New incident reported",
            latitude=float(data.get("latitude", world.crisis.latitude)),
            longitude=float(data.get("longitude", world.crisis.longitude)),
            severity=float(data.get("severity", 5)),
            timestamp=datetime.fromisoformat(str(data["timestamp"]).replace("Z", "+00:00"))
            if data.get("timestamp")
            else datetime.fromisoformat(world.crisis.current_time.isoformat()),
            zone_id=data.get("zone_id") or next(iter(world.zones)),
            status=IncidentStatus(data.get("status", "active")),
            source=data.get("source", "operator"),
            confidence=float(data.get("confidence", 0.7)),
        )
        if incident.zone_id not in world.zones:
            raise InvalidEventError("incident zone_id is invalid")
        world.incidents.append(incident)
        return incident.model_dump(mode="json")

    def _set_road(self, world: CrisisWorld, payload: dict[str, Any], status: RoadStatus) -> dict[str, Any]:
        road_id = payload.get("target_id") or payload.get("road_id")
        if road_id not in world.roads:
            raise InvalidEventError("valid road_id is required")
        world.roads[road_id].status = status
        if status == RoadStatus.OPEN:
            world.roads[road_id].risk_score = min(world.roads[road_id].risk_score, 30)
        if status == RoadStatus.BLOCKED:
            world.roads[road_id].risk_score = max(world.roads[road_id].risk_score, 85)
        return world.roads[road_id].model_dump(mode="json")

    def _event_hospital(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        hid = payload.get("target_id") or payload.get("hospital_id")
        if hid not in world.hospitals:
            raise InvalidEventError("valid hospital_id is required")
        hospital = world.hospitals[hid]
        if "available_beds" in payload:
            beds = int(payload["available_beds"])
            if beds < 0:
                raise InvalidEventError("available_beds cannot be negative")
            hospital.available_beds = min(hospital.total_beds, beds)
        if "delta_beds" in payload:
            hospital.available_beds = max(0, min(hospital.total_beds, hospital.available_beds + int(payload["delta_beds"])))
        if hospital.available_beds == 0:
            hospital.status = HospitalStatus.LIMITED
        return hospital.model_dump(mode="json")

    def _event_shelter(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        sid = payload.get("target_id") or payload.get("shelter_id")
        if sid not in world.shelters:
            raise InvalidEventError("valid shelter_id is required")
        shelter = world.shelters[sid]
        if "current_occupancy" in payload:
            occ = int(payload["current_occupancy"])
            if occ < 0:
                raise InvalidEventError("occupancy cannot be negative")
            shelter.current_occupancy = min(shelter.capacity, occ)
        if payload.get("full"):
            shelter.current_occupancy = shelter.capacity
            shelter.status = ShelterStatus.FULL
        return shelter.model_dump(mode="json")

    def _event_resource(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        rid = payload.get("target_id") or payload.get("resource_id")
        if rid not in world.resources:
            raise InvalidEventError("valid resource_id is required")
        resource = world.resources[rid]
        if "available_quantity" in payload:
            qty = int(payload["available_quantity"])
            if qty < 0:
                raise InvalidEventError("available_quantity cannot be negative")
            resource.available_quantity = min(resource.quantity, qty)
        if "delta" in payload:
            resource.available_quantity = max(0, min(resource.quantity, resource.available_quantity + int(payload["delta"])))
        if payload.get("status"):
            resource.status = ResourceStatus(payload["status"])
        return resource.model_dump(mode="json")

    def _event_water(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        params = world.crisis.disaster_parameters
        if "value" in payload:
            params["river_water_level_m"] = float(payload["value"])
        if "delta" in payload:
            params["river_water_level_m"] = float(params.get("river_water_level_m", 0)) + float(payload["delta"])
            params["water_level_change_m"] = float(payload["delta"])
        zone_id = payload.get("target_id") or payload.get("zone_id")
        if zone_id in world.zones:
            if "value" in payload:
                world.zones[zone_id].parameters["water_level_m"] = float(payload["value"])
            if "delta" in payload:
                current = float(world.zones[zone_id].parameters.get("water_level_m", 0))
                world.zones[zone_id].parameters["water_level_m"] = max(0, current + float(payload["delta"]))
                world.zones[zone_id].parameters["water_level_change_m"] = float(payload["delta"])
        return {"disaster_parameters": params}

    def _event_rain(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        params = world.crisis.disaster_parameters
        if "value" in payload:
            params["rainfall_mm"] = float(payload["value"])
        if "delta_percent" in payload:
            params["rainfall_mm"] = float(params.get("rainfall_mm", 0)) * (1 + float(payload["delta_percent"]) / 100)
        if "delta" in payload:
            params["rainfall_mm"] = float(params.get("rainfall_mm", 0)) + float(payload["delta"])
        for zone in world.zones.values():
            rain = float(zone.parameters.get("rainfall_mm", params.get("rainfall_mm", 0)))
            if "delta_percent" in payload:
                zone.parameters["rainfall_mm"] = rain * (1 + float(payload["delta_percent"]) / 100)
            elif "delta" in payload:
                zone.parameters["rainfall_mm"] = rain + float(payload["delta"])
        return {"rainfall_mm": params.get("rainfall_mm")}

    def _event_wind(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        params = world.crisis.disaster_parameters
        if "value" in payload:
            params["wind_speed_kmh"] = float(payload["value"])
        if "delta_percent" in payload:
            params["wind_speed_kmh"] = float(params.get("wind_speed_kmh", 0)) * (1 + float(payload["delta_percent"]) / 100)
            params["intensity_index"] = min(1.0, float(params.get("intensity_index", 0.5)) * (1 + float(payload["delta_percent"]) / 200))
        for zone in world.zones.values():
            wind = float(zone.parameters.get("wind_speed_kmh", params.get("wind_speed_kmh", 0)))
            if "delta_percent" in payload:
                zone.parameters["wind_speed_kmh"] = wind * (1 + float(payload["delta_percent"]) / 100)
            elif "value" in payload:
                zone.parameters["wind_speed_kmh"] = float(payload["value"])
        return {"wind_speed_kmh": params.get("wind_speed_kmh")}

    def _event_quake(self, world: CrisisWorld, payload: dict[str, Any]) -> dict[str, Any]:
        params = world.crisis.disaster_parameters
        if "magnitude" in payload:
            params["magnitude"] = float(payload["magnitude"])
        if "delta_magnitude" in payload:
            params["magnitude"] = float(params.get("magnitude", 0)) + float(payload["delta_magnitude"])
        if "seismic_intensity" in payload and payload.get("target_id") in world.zones:
            world.zones[payload["target_id"]].parameters["seismic_intensity"] = float(payload["seismic_intensity"])
        return {"magnitude": params.get("magnitude")}


crisis_service = CrisisService()
