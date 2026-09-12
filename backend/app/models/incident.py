from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    ACTIVE = "active"
    CONTAINED = "contained"
    RESOLVED = "resolved"


class Incident(BaseModel):
    id: str
    crisis_id: str
    type: str
    description: str
    latitude: float
    longitude: float
    severity: float = Field(ge=0, le=10)
    timestamp: datetime
    zone_id: str
    status: IncidentStatus = IncidentStatus.ACTIVE
    source: str = "sensor_network"
    confidence: float = Field(ge=0, le=1, default=0.8)
