from __future__ import annotations

"""Archived Legacy Crisis Service.

This service is part of the deprecated synthetic Harbor City prototype.
It is retained under app/legacy strictly for migration reference and
is NOT used in the DisasterPulse TN production flood runtime.
"""

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


class LegacyCrisisService:
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
            try:
                apply_risk_to_world(world)
            except RuntimeError as exc:
                if world.crisis.disaster_type == DisasterType.FLOOD and "No trained flood model" in str(exc):
                    logger.warning("Flood ML model is not trained yet; flood risks will be computed after training.")
                    continue
                raise
        logger.info("Loaded legacy crisis worlds for flood, cyclone, and earthquake")

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


legacy_crisis_service = LegacyCrisisService()
