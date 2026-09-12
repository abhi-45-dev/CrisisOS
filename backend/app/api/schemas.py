from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.response import PlanAction, ResponsePlan, ScenarioChange


class EventRequest(BaseModel):
    type: str = Field(description="Event type such as road_closure or rainfall_change")
    disaster_type: str = Field(default="flood")
    target_id: str | None = None
    zone_id: str | None = None
    road_id: str | None = None
    hospital_id: str | None = None
    shelter_id: str | None = None
    resource_id: str | None = None
    value: float | None = None
    delta: float | None = None
    delta_percent: float | None = None
    delta_beds: int | None = None
    available_beds: int | None = None
    current_occupancy: int | None = None
    available_quantity: int | None = None
    full: bool | None = None
    status: str | None = None
    magnitude: float | None = None
    delta_magnitude: float | None = None
    seismic_intensity: float | None = None
    readings: dict[str, Any] | None = None
    incident: dict[str, Any] | None = None
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    severity: float | None = None
    confidence: float | None = None
    timestamp: datetime | None = None
    crisis_id: str | None = None


class SimulateRequest(BaseModel):
    disaster_type: str = "flood"
    plan: ResponsePlan | None = None
    actions: list[PlanAction] = Field(default_factory=list)
    title: str = "Operator response plan"
    objective: str = "Reduce risk and evacuate affected populations"


class CompareRequest(BaseModel):
    disaster_type: str = "flood"
    plans: list[ResponsePlan]


class AnalyzeRequest(BaseModel):
    disaster_type: str = "flood"
    question: str | None = "What are the most critical areas right now?"


class PlanRequest(BaseModel):
    disaster_type: str = "flood"
    objective: str | None = "Create an evacuation plan."


class WhatIfRequest(BaseModel):
    disaster_type: str = "flood"
    question: str


class ScenarioCreateRequest(BaseModel):
    disaster_type: str = "flood"
    description: str
    changes: list[ScenarioChange]


class RouteQuery(BaseModel):
    origin: str
    destination: str
    disaster_type: str = "flood"


class GeoQuery(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 5
    disaster_type: str = "flood"
    resource_type: str | None = None
