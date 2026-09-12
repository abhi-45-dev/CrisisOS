from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DisasterType(str, Enum):
    FLOOD = "flood"
    CYCLONE = "cyclone"
    EARTHQUAKE = "earthquake"


class CrisisStatus(str, Enum):
    MONITORING = "monitoring"
    ACTIVE = "active"
    ESCALATING = "escalating"
    RECOVERY = "recovery"
    CONTAINED = "contained"


class Crisis(BaseModel):
    id: str
    name: str
    disaster_type: DisasterType
    status: CrisisStatus = CrisisStatus.ACTIVE
    start_time: datetime
    current_time: datetime
    severity: float = Field(ge=0, le=10)
    description: str
    disaster_parameters: dict[str, Any] = Field(default_factory=dict)
    region_name: str = "Tamil Nadu"
    latitude: float = 13.0827
    longitude: float = 80.2707
