from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.location import RiskLevel


class PlanAction(BaseModel):
    type: str
    target_id: str | None = None
    quantity: int | float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ResponsePlan(BaseModel):
    id: str = ""
    title: str
    objective: str
    actions: list[PlanAction] = Field(default_factory=list)
    affected_zones: list[str] = Field(default_factory=list)
    assigned_resources: list[str] = Field(default_factory=list)
    evacuation_routes: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    estimated_people_helped: int = 0
    estimated_time: float = 0
    estimated_risk_reduction: float = 0
    expected_outcomes: list[str] = Field(default_factory=list)
    score: float = 0
    confidence: float = Field(ge=0, le=1, default=0.6)
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ScenarioChange(BaseModel):
    type: str
    target_id: str | None = None
    field: str | None = None
    value: Any = None
    delta: float | None = None
    delta_percent: float | None = None
    new_status: str | None = None
    description: str = ""


class Scenario(BaseModel):
    id: str
    crisis_id: str
    description: str
    changes: list[ScenarioChange] = Field(default_factory=list)
    created_at: datetime
    resulting_state: dict[str, Any] | None = None
    simulation_result: dict[str, Any] | None = None


class RiskResult(BaseModel):
    zone_id: str
    zone_name: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    contributing_factors: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    disaster_type: str
    population: int = 0
    affected: bool = False


class RouteLeg(BaseModel):
    road_id: str
    road_name: str
    from_zone: str
    to_zone: str
    distance_km: float
    travel_time_min: float
    status: str


class RouteResult(BaseModel):
    origin: str
    destination: str
    path_zones: list[str] = Field(default_factory=list)
    roads_used: list[str] = Field(default_factory=list)
    legs: list[RouteLeg] = Field(default_factory=list)
    total_distance_km: float = 0
    estimated_travel_time_min: float = 0
    route_risk: float = 0
    warnings: list[str] = Field(default_factory=list)
    feasible: bool = True


class SimulationResult(BaseModel):
    disaster_type: str
    people_requiring_evacuation: int = 0
    people_evacuated: int = 0
    people_remaining: int = 0
    people_protected: int = 0
    shelter_utilization: float = 0
    evacuation_time_min: float = 0
    evacuation_completion: float = 0
    risk_reduction: float = 0
    response_time_min: float = 0
    resource_utilization: float = 0
    allocation_score: float = 0
    overall_score: float = 0
    bottlenecks: list[str] = Field(default_factory=list)
    unsafe_routes: list[str] = Field(default_factory=list)
    unmet_demand: dict[str, Any] = Field(default_factory=dict)
    assignments: list[dict[str, Any]] = Field(default_factory=list)
    zone_results: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class ClaimBucket(BaseModel):
    observed_facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AIAnalysis(BaseModel):
    summary: str
    observations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)
    assumptions: list[str] = Field(default_factory=list)
    critical_zones: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    claim_buckets: ClaimBucket = Field(default_factory=ClaimBucket)
    llm_available: bool = True
    mode: str = "llm"


class Critique(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)
    claim_buckets: ClaimBucket = Field(default_factory=ClaimBucket)
    llm_available: bool = True
    mode: str = "llm"


class WhatIfResult(BaseModel):
    scenario: Scenario
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    changes: list[ScenarioChange] = Field(default_factory=list)
    impacts: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.55)
    affected_zones: list[str] = Field(default_factory=list)
    changed_routes: list[dict[str, Any]] = Field(default_factory=list)
    changed_risk: list[dict[str, Any]] = Field(default_factory=list)
    changed_resource_allocation: dict[str, Any] = Field(default_factory=dict)
    changed_evacuation: dict[str, Any] = Field(default_factory=dict)
    impact_summary: str = ""
    llm_available: bool = True
    mode: str = "llm"
    claim_buckets: ClaimBucket = Field(default_factory=ClaimBucket)


class WhatIfExplanation(BaseModel):
    impact_summary: str = ""
    impacts: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.55)
    claim_buckets: ClaimBucket = Field(default_factory=ClaimBucket)
