from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ZoneEvacuationStatus(str, Enum):
    NONE = "none"
    ADVISED = "advised"
    ORDERED = "ordered"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class RoadStatus(str, Enum):
    OPEN = "open"
    CONGESTED = "congested"
    BLOCKED = "blocked"
    DANGEROUS = "dangerous"


class HospitalStatus(str, Enum):
    OPERATIONAL = "operational"
    STRAINED = "strained"
    LIMITED = "limited"
    OFFLINE = "offline"


class ShelterStatus(str, Enum):
    OPEN = "open"
    FILLING = "filling"
    FULL = "full"
    CLOSED = "closed"


class GeoPoint(BaseModel):
    latitude: float
    longitude: float


class Zone(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    population: int = Field(ge=0)
    vulnerability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=100, default=0)
    risk_level: RiskLevel = RiskLevel.LOW
    affected: bool = False
    evacuation_status: ZoneEvacuationStatus = ZoneEvacuationStatus.NONE
    parameters: dict[str, Any] = Field(default_factory=dict)


class Road(BaseModel):
    id: str
    name: str
    from_zone: str
    to_zone: str
    distance_km: float = Field(gt=0, alias="distance")
    travel_time_min: float = Field(gt=0, alias="travel_time")
    capacity: int = Field(ge=0)
    status: RoadStatus = RoadStatus.OPEN
    risk_score: float = Field(ge=0, le=100, default=0)
    latitude: float
    longitude: float
    path: list[list[float]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def distance(self) -> float:
        return self.distance_km

    @property
    def travel_time(self) -> float:
        return self.travel_time_min


class Hospital(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    zone_id: str
    total_beds: int = Field(ge=0)
    available_beds: int = Field(ge=0)
    emergency_capacity: int = Field(ge=0)
    status: HospitalStatus = HospitalStatus.OPERATIONAL

    @model_validator(mode="after")
    def clamp_beds(self) -> "Hospital":
        if self.available_beds > self.total_beds:
            self.available_beds = self.total_beds
        if self.available_beds < 0:
            self.available_beds = 0
        return self


class Shelter(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    zone_id: str
    capacity: int = Field(ge=0)
    current_occupancy: int = Field(ge=0)
    status: ShelterStatus = ShelterStatus.OPEN

    @computed_field
    @property
    def available_capacity(self) -> int:
        return max(0, self.capacity - self.current_occupancy)

    @model_validator(mode="after")
    def clamp_occupancy(self) -> "Shelter":
        if self.current_occupancy < 0:
            self.current_occupancy = 0
        if self.current_occupancy > self.capacity:
            self.current_occupancy = self.capacity
        if self.available_capacity <= 0:
            self.status = ShelterStatus.FULL
        elif self.current_occupancy / max(self.capacity, 1) >= 0.8:
            self.status = ShelterStatus.FILLING
        return self


class Alert(BaseModel):
    id: str
    severity: RiskLevel
    type: str
    message: str
    timestamp: str
    affected_zone: str | None = None
    evidence: list[str] = Field(default_factory=list)
    disaster_type: str | None = None
