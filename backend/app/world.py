from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.models.crisis import Crisis
from app.models.incident import Incident
from app.models.location import Hospital, Road, Shelter, Zone
from app.models.resource import Resource
from app.models.response import Scenario


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CrisisWorld:
    crisis: Crisis
    zones: dict[str, Zone]
    roads: dict[str, Road]
    hospitals: dict[str, Hospital]
    shelters: dict[str, Shelter]
    resources: dict[str, Resource]
    incidents: list[Incident] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)

    def clone(self) -> "CrisisWorld":
        return deepcopy(self)

    def snapshot(self) -> dict[str, Any]:
        return {
            "crisis": self.crisis.model_dump(mode="json"),
            "zones": [z.model_dump(mode="json") for z in self.zones.values()],
            "roads": [r.model_dump(mode="json") for r in self.roads.values()],
            "hospitals": [h.model_dump(mode="json") for h in self.hospitals.values()],
            "shelters": [s.model_dump(mode="json") for s in self.shelters.values()],
            "resources": [r.model_dump(mode="json") for r in self.resources.values()],
            "incidents": [i.model_dump(mode="json") for i in self.incidents],
        }
