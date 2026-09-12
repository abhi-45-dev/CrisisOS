from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ResourceType(str, Enum):
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    RESCUE_TEAM = "rescue_team"
    POLICE_UNIT = "police_unit"
    HELICOPTER = "helicopter"
    WATER_PUMP = "water_pump"
    MEDICAL_TEAM = "medical_team"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    FOOD_SUPPLY = "food_supply"


class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    DEPLOYED = "deployed"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


class Resource(BaseModel):
    id: str
    type: ResourceType
    name: str
    quantity: int = Field(ge=0)
    available_quantity: int = Field(ge=0)
    latitude: float
    longitude: float
    zone_id: str
    status: ResourceStatus = ResourceStatus.AVAILABLE

    @model_validator(mode="after")
    def clamp_quantity(self) -> "Resource":
        if self.available_quantity < 0:
            self.available_quantity = 0
        if self.available_quantity > self.quantity:
            self.available_quantity = self.quantity
        if self.available_quantity == 0:
            self.status = ResourceStatus.UNAVAILABLE
        elif self.available_quantity < self.quantity:
            self.status = ResourceStatus.LIMITED
        return self
