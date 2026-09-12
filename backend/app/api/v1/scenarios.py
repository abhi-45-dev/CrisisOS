from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.scenario_service import scenario_service

router = APIRouter(prefix="/scenarios")


class RainfallScenarioRequest(BaseModel):
    rainfall_delta_mm: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Rainfall increase in millimeters to add to feature vectors",
    )
    evaluate_route_origin: list[float] | None = Field(
        default=None,
        description="Optional [latitude, longitude] origin to test route sensitivity",
    )
    evaluate_route_destination: list[float] | None = Field(
        default=None,
        description="Optional [latitude, longitude] destination to test route sensitivity",
    )


@router.post("/rainfall")
def simulate_rainfall_scenario(body: RainfallScenarioRequest):
    """
    Run genuine FloodNow TN ML model inference for a what-if rainfall delta.
    
    Reruns the trained classifier with the modified rainfall feature vector,
    returning baseline vs scenario risk bands, probability deltas, and cell escalations.
    """
    try:
        return scenario_service.simulate_rainfall_delta(
            rainfall_delta_mm=body.rainfall_delta_mm,
            evaluate_route_origin=body.evaluate_route_origin,
            evaluate_route_destination=body.evaluate_route_destination,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario simulation failed: {str(e)}",
        )
